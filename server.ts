import 'dotenv/config';
import crypto from 'crypto';
import fs from 'fs/promises';
import http from 'http';
import path from 'path';
import zlib from 'zlib';
import express, { type NextFunction, type Request, type Response } from 'express';
import { GoogleGenAI, Type } from '@google/genai';
import { createServer as createViteServer } from 'vite';
import {
  ACHIEVEMENTS,
  DEFAULT_APPS,
  PERSONA_DETAILS,
  RANK_TIERS,
  SELF_IMPROVEMENT_GOALS,
  computeLevelInfo,
} from './src/shared/catalog.ts';
import type {
  AccentPalette,
  ActiveLock,
  AppCatalogItem,
  AuthProvider,
  CritiqueEvaluation,
  CritiqueHistoryEntry,
  CritiquePersona,
  CustomizationSettings,
  DailyQuest,
  LeaderboardEntry,
  RankTierName,
  RoutineStep,
  SelfImprovementGoal,
  ThemeMode,
  UserRoutine,
  UserState,
} from './src/shared/types.ts';
import { validateEmailStrict, validatePasswordStrength } from './src/shared/validation.ts';

interface StoredUser {
  id: string;
  email: string;
  name: string;
  authProvider?: AuthProvider;
  oauthProviderId?: string;
  passwordHash: string;
  salt: string;
  createdAt: string;
  xp: number;
  streakDays: number;
  streakFreezes?: number;
  checkInsCompleted?: number;
  lastActiveDate: string | null;
  focusCoins: number;
  totalMinutesLocked: number;
  totalTasksPassed: number;
  bestScore: number;
  settings: CustomizationSettings;
  customApps: AppCatalogItem[];
  activeLocks: ActiveLock[];
  routines?: UserRoutine[];
  lastRoutineResetDate?: string;
  rewardedRoutineStepKeys?: string[];
  critiqueHistory: CritiqueHistoryEntry[];
  unlockedAchievements: string[];
  claimedQuestKeys: string[];
  rememberTokens?: Array<{ hash: string; expiresAt: number }>;
}

interface DatabaseSchema {
  users: Record<string, StoredUser>;
}

const PORT = Number(process.env.PORT || 3000);
const DATA_DIR = path.resolve(process.cwd(), '.data');
const DB_PATH = path.join(DATA_DIR, 'critique-db.json');
const SESSION_SECRET = process.env.SESSION_SECRET || crypto.randomBytes(32).toString('hex');
const SESSION_TTL_MS = 7 * 24 * 60 * 60 * 1000;
const REMEMBER_COOKIE_NAME = 'enhance_session';
const REMEMBER_TTL_MS = 30 * 24 * 60 * 60 * 1000;

const revokedTokens = new Set<string>();
const loginAttempts = new Map<string, { count: number; resetAt: number }>();

function hashRememberToken(rawToken: string): string {
  return crypto.createHash('sha256').update(rawToken).digest('hex');
}

function parseCookies(cookieHeader: string | undefined): Record<string, string> {
  const out: Record<string, string> = {};
  if (!cookieHeader) return out;
  for (const part of cookieHeader.split(';')) {
    const idx = part.indexOf('=');
    if (idx === -1) continue;
    const key = part.slice(0, idx).trim();
    const val = part.slice(idx + 1).trim();
    if (key) {
      try {
        out[key] = decodeURIComponent(val);
      } catch {
        out[key] = val;
      }
    }
  }
  return out;
}

function isSecureRequest(req: Request): boolean {
  const forwardedProto = req.get('x-forwarded-proto')?.split(',')[0].trim();
  return req.secure || forwardedProto === 'https';
}

function issueRememberCookie(req: Request, res: Response, user: StoredUser): void {
  const rawToken = crypto.randomBytes(36).toString('base64url');
  const hash = hashRememberToken(rawToken);
  const now = Date.now();
  const active = (user.rememberTokens ?? []).filter((entry) => entry.expiresAt > now);
  active.push({ hash, expiresAt: now + REMEMBER_TTL_MS });
  user.rememberTokens = active.slice(-10);

  const isHttps = isSecureRequest(req);
  const maxAgeSec = Math.floor(REMEMBER_TTL_MS / 1000);
  const cookieParts = [
    `${REMEMBER_COOKIE_NAME}=${encodeURIComponent(rawToken)}`,
    'Path=/',
    `Max-Age=${maxAgeSec}`,
    'HttpOnly',
    ...(isHttps ? ['Secure', 'SameSite=None', 'Partitioned'] : ['SameSite=Lax']),
  ];
  res.append('Set-Cookie', cookieParts.join('; '));
}

function clearRememberCookie(req: Request, res: Response): void {
  const isHttps = isSecureRequest(req);
  const cookieParts = [
    `${REMEMBER_COOKIE_NAME}=`,
    'Path=/',
    'Max-Age=0',
    'HttpOnly',
    ...(isHttps ? ['Secure', 'SameSite=None', 'Partitioned'] : ['SameSite=Lax']),
  ];
  res.append('Set-Cookie', cookieParts.join('; '));
}

type SocialOAuthProvider = 'facebook' | 'instagram' | 'tiktok';

interface PendingOAuthState {
  provider: SocialOAuthProvider;
  expiresAt: number;
  preAuth?: {
    age?: number;
    selectedGoals?: SelfImprovementGoal[];
    accent?: AccentPalette;
    critiquePersona?: CritiquePersona;
    theme?: ThemeMode;
  };
}

interface PendingOAuthTicket {
  email: string;
  provider: SocialOAuthProvider;
  expiresAt: number;
}

const oauthStates = new Map<string, PendingOAuthState>();
const oauthTickets = new Map<string, PendingOAuthTicket>();

function pruneOAuthMaps(): void {
  const now = Date.now();
  for (const [k, v] of oauthStates) {
    if (now > v.expiresAt) oauthStates.delete(k);
  }
  for (const [k, v] of oauthTickets) {
    if (now > v.expiresAt) oauthTickets.delete(k);
  }
}

function getAppBaseUrl(): string {
  const raw = process.env.APP_URL?.trim();
  if (raw) {
    return raw.replace(/\/+$/, '');
  }
  return `http://localhost:${PORT}`;
}

function getProviderConfig(provider: SocialOAuthProvider): {
  configured: boolean;
  clientId: string;
  clientSecret: string;
  callbackUrl: string;
  requiredVars: string[];
} {
  const callbackUrl = `${getAppBaseUrl()}/auth/oauth/${provider}/callback`;
  if (provider === 'facebook') {
    const clientId = process.env.FACEBOOK_CLIENT_ID?.trim() || '';
    const clientSecret = process.env.FACEBOOK_CLIENT_SECRET?.trim() || '';
    return {
      configured: Boolean(clientId && clientSecret),
      clientId,
      clientSecret,
      callbackUrl,
      requiredVars: ['FACEBOOK_CLIENT_ID', 'FACEBOOK_CLIENT_SECRET'],
    };
  }
  if (provider === 'instagram') {
    const clientId = process.env.INSTAGRAM_CLIENT_ID?.trim() || '';
    const clientSecret = process.env.INSTAGRAM_CLIENT_SECRET?.trim() || '';
    return {
      configured: Boolean(clientId && clientSecret),
      clientId,
      clientSecret,
      callbackUrl,
      requiredVars: ['INSTAGRAM_CLIENT_ID', 'INSTAGRAM_CLIENT_SECRET'],
    };
  }
  const clientId = process.env.TIKTOK_CLIENT_KEY?.trim() || '';
  const clientSecret = process.env.TIKTOK_CLIENT_SECRET?.trim() || '';
  return {
    configured: Boolean(clientId && clientSecret),
    clientId,
    clientSecret,
    callbackUrl,
    requiredVars: ['TIKTOK_CLIENT_KEY', 'TIKTOK_CLIENT_SECRET'],
  };
}

let writeChain: Promise<void> = Promise.resolve();
let dbCache: DatabaseSchema | null = null;

async function loadDb(): Promise<DatabaseSchema> {
  if (dbCache) return dbCache;
  try {
    await fs.mkdir(DATA_DIR, { recursive: true });
    const raw = await fs.readFile(DB_PATH, 'utf8');
    const parsed = JSON.parse(raw) as DatabaseSchema;
    if (parsed && typeof parsed === 'object' && parsed.users) {
      dbCache = parsed;
      return dbCache;
    }
  } catch {
    // Initialize fresh DB when absent
  }
  dbCache = { users: {} };
  await persistDb(dbCache);
  return dbCache;
}

async function persistDb(db: DatabaseSchema): Promise<void> {
  dbCache = db;
  writeChain = writeChain.then(async () => {
    await fs.mkdir(DATA_DIR, { recursive: true });
    const tmpPath = `${DB_PATH}.${crypto.randomUUID()}.tmp`;
    await fs.writeFile(tmpPath, JSON.stringify(db, null, 2), 'utf8');
    await fs.rename(tmpPath, DB_PATH);
  });
  return writeChain;
}

function hashPassword(password: string, salt?: string): { hash: string; salt: string } {
  const actualSalt = salt || crypto.randomBytes(16).toString('hex');
  const derived = crypto.scryptSync(password, actualSalt, 64).toString('hex');
  return { hash: derived, salt: actualSalt };
}

function verifyPassword(password: string, storedHash: string, salt: string): boolean {
  const derived = crypto.scryptSync(password, salt, 64);
  const storedBuf = Buffer.from(storedHash, 'hex');
  if (derived.length !== storedBuf.length) return false;
  return crypto.timingSafeEqual(derived, storedBuf);
}

interface TokenPayload {
  sub: string;
  email: string;
  jti: string;
  exp: number;
}

function signSessionToken(user: StoredUser): string {
  const payload: TokenPayload = {
    sub: user.id,
    email: user.email,
    jti: crypto.randomUUID(),
    exp: Date.now() + SESSION_TTL_MS,
  };
  const encodedPayload = Buffer.from(JSON.stringify(payload), 'utf8').toString('base64url');
  const signature = crypto
    .createHmac('sha256', SESSION_SECRET)
    .update(encodedPayload)
    .digest('base64url');
  return `${encodedPayload}.${signature}`;
}

function verifySessionToken(token: string): TokenPayload | null {
  const parts = token.split('.');
  if (parts.length !== 2) return null;
  const [encodedPayload, signature] = parts;
  const expectedSig = crypto
    .createHmac('sha256', SESSION_SECRET)
    .update(encodedPayload)
    .digest('base64url');

  const sigBuf = Buffer.from(signature, 'utf8');
  const expBuf = Buffer.from(expectedSig, 'utf8');
  if (sigBuf.length !== expBuf.length || !crypto.timingSafeEqual(sigBuf, expBuf)) {
    return null;
  }

  try {
    const payload = JSON.parse(
      Buffer.from(encodedPayload, 'base64url').toString('utf8')
    ) as TokenPayload;
    if (!payload || !payload.email || !payload.sub || !payload.jti) return null;
    if (Date.now() > payload.exp) return null;
    if (revokedTokens.has(payload.jti)) return null;
    return payload;
  } catch {
    return null;
  }
}

function todayKey(): string {
  return new Date().toISOString().slice(0, 10);
}

function yesterdayKey(): string {
  return new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString().slice(0, 10);
}

function touchStreak(user: StoredUser): void {
  const today = todayKey();
  if (user.lastActiveDate === today) return;
  if (user.lastActiveDate === yesterdayKey()) {
    user.streakDays += 1;
  } else if (user.lastActiveDate && (user.streakFreezes ?? 0) > 0 && user.streakDays > 0) {
    user.streakFreezes = (user.streakFreezes ?? 1) - 1;
    user.streakDays += 1;
  } else {
    user.streakDays = 1;
  }
  user.lastActiveDate = today;
}

function awardAchievement(user: StoredUser, achievementId: string): void {
  if (user.unlockedAchievements.includes(achievementId)) return;
  const def = ACHIEVEMENTS.find((a) => a.id === achievementId);
  if (!def) return;
  user.unlockedAchievements.push(achievementId);
  user.xp += def.xpBonus;
}

function computeDailyQuests(user: StoredUser): DailyQuest[] {
  const today = todayKey();
  const locksToday = user.activeLocks.filter((l) => l.lockedAt.startsWith(today)).length;
  const passedToday = user.critiqueHistory.filter(
    (h) => h.createdAt.startsWith(today) && h.evaluation.passed
  );
  const highScoreToday = passedToday.some((h) => h.evaluation.score >= 85);

  const q1Key = `${today}:lock-1`;
  const q2Key = `${today}:pass-1`;
  const q3Key = `${today}:score-85`;

  if (locksToday >= 1 && !user.claimedQuestKeys.includes(q1Key)) {
    user.claimedQuestKeys.push(q1Key);
    user.xp += 60;
    user.focusCoins += 15;
  }
  if (passedToday.length >= 1 && !user.claimedQuestKeys.includes(q2Key)) {
    user.claimedQuestKeys.push(q2Key);
    user.xp += 120;
    user.focusCoins += 30;
  }
  if (highScoreToday && !user.claimedQuestKeys.includes(q3Key)) {
    user.claimedQuestKeys.push(q3Key);
    user.xp += 150;
    user.focusCoins += 45;
  }

  return [
    {
      id: 'quest-lock-1',
      title: 'Lock In Your Attention',
      description: 'Start at least 1 timed app lock today.',
      xpReward: 60,
      coinsReward: 15,
      completed: locksToday >= 1,
      progressText: `${Math.min(locksToday, 1)} / 1 lock`,
    },
    {
      id: 'quest-pass-1',
      title: 'Earned Unlock',
      description: 'Complete and pass 1 AI task critique today.',
      xpReward: 120,
      coinsReward: 30,
      completed: passedToday.length >= 1,
      progressText: `${Math.min(passedToday.length, 1)} / 1 passed`,
    },
    {
      id: 'quest-score-85',
      title: 'High-Bar Precision',
      description: 'Achieve a score of 85 or above on a task critique.',
      xpReward: 150,
      coinsReward: 45,
      completed: highScoreToday,
      progressText: highScoreToday ? 'Achieved today' : '0 / 1 (85+ score)',
    },
  ];
}

function syncExpiredLocks(user: StoredUser): boolean {
  const now = Date.now();
  let changed = false;
  for (const lock of user.activeLocks) {
    if (lock.status === 'locked' && new Date(lock.unlocksAt).getTime() <= now) {
      lock.status = 'expired_completed';
      user.xp += Math.max(20, Math.round(lock.durationMinutes * 1.2));
      user.focusCoins += Math.max(10, Math.round(lock.durationMinutes / 3));
      user.totalMinutesLocked += lock.durationMinutes;
      touchStreak(user);
      changed = true;
    }
  }
  return changed;
}

function syncDailyRoutineReset(user: StoredUser): boolean {
  const today = todayKey();
  if (!Array.isArray(user.routines)) {
    user.routines = [];
    user.lastRoutineResetDate = today;
    return true;
  }
  if (user.lastRoutineResetDate === today) return false;
  for (const routine of user.routines) {
    for (const step of routine.steps) {
      step.completedToday = false;
    }
  }
  user.lastRoutineResetDate = today;
  return true;
}

const VALID_GOALS = new Set<SelfImprovementGoal>([
  'fitness',
  'looks_style',
  'social_skills',
  'mindset',
]);

function normalizeSelectedGoals(
  rawSelected: unknown,
  fallbackGoal: SelfImprovementGoal = 'mindset'
): SelfImprovementGoal[] {
  const result: SelfImprovementGoal[] = [];
  if (Array.isArray(rawSelected)) {
    for (const g of rawSelected) {
      if (VALID_GOALS.has(g as SelfImprovementGoal) && !result.includes(g as SelfImprovementGoal)) {
        result.push(g as SelfImprovementGoal);
      }
    }
  }
  if (result.length === 0) {
    result.push(VALID_GOALS.has(fallbackGoal) ? fallbackGoal : 'mindset');
  }
  return result;
}

function sanitizeUser(user: StoredUser): UserState {
  const quests = computeDailyQuests(user);
  const levelInfo = computeLevelInfo(user.xp);
  const fallbackGoal = VALID_GOALS.has(user.settings.mainGoal) ? user.settings.mainGoal : 'mindset';
  const selectedGoals = normalizeSelectedGoals(user.settings.selectedGoals, fallbackGoal);
  const safeSettings: CustomizationSettings = {
    theme: user.settings.theme ?? 'dark',
    accent: user.settings.accent ?? 'indigo',
    critiquePersona: user.settings.critiquePersona ?? 'balanced',
    hardcoreMode: Boolean(user.settings.hardcoreMode),
    soundEnabled: user.settings.soundEnabled ?? true,
    pushNotificationsEnabled: Boolean(user.settings.pushNotificationsEnabled),
    density: user.settings.density ?? 'comfortable',
    displayName: user.settings.displayName || user.name,
    age: typeof user.settings.age === 'number' ? user.settings.age : null,
    mainGoal: selectedGoals[0],
    selectedGoals,
  };
  return {
    id: user.id,
    email: user.email,
    name: safeSettings.displayName || user.name,
    authProvider: user.authProvider ?? 'email',
    xp: user.xp,
    level: levelInfo.level,
    rankTitle: levelInfo.rankTitle,
    nextRankTitle: levelInfo.nextRankTitle,
    xpIntoLevel: levelInfo.xpIntoLevel,
    xpForNextLevel: levelInfo.xpForNextLevel,
    streakDays: user.streakDays,
    streakFreezes: user.streakFreezes ?? 0,
    checkInsCompleted: user.checkInsCompleted ?? 0,
    focusCoins: user.focusCoins,
    totalMinutesLocked: user.totalMinutesLocked,
    totalTasksPassed: user.totalTasksPassed,
    bestScore: user.bestScore,
    settings: safeSettings,
    customApps: user.customApps,
    activeLocks: user.activeLocks,
    routines: user.routines ?? [],
    critiqueHistory: user.critiqueHistory.slice(0, 40),
    unlockedAchievements: user.unlockedAchievements,
    quests,
  };
}

interface AuthedRequest extends Request {
  authPayload?: TokenPayload;
}

function requireAuth(req: AuthedRequest, res: Response, next: NextFunction): void {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    res.status(401).json({ error: 'Authentication required. Please sign in.' });
    return;
  }
  const token = authHeader.slice('Bearer '.length).trim();
  const payload = verifySessionToken(token);
  if (!payload) {
    res.status(401).json({ error: 'Your session has expired. Please sign in again.' });
    return;
  }
  req.authPayload = payload;
  next();
}

function createGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY is not configured in server secrets.');
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

// Lightweight CRC32 & ZIP archive builder (no external zip binary needed)
const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[i] = c >>> 0;
  }
  return table;
})();

function crc32(buf: Buffer): number {
  let crc = 0xffffffff;
  for (let i = 0; i < buf.length; i++) {
    crc = CRC_TABLE[(crc ^ buf[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function buildZipBuffer(entries: Array<{ name: string; data: Buffer }>): Buffer {
  const localParts: Buffer[] = [];
  const centralParts: Buffer[] = [];
  let offset = 0;

  for (const entry of entries) {
    const nameBuf = Buffer.from(entry.name, 'utf8');
    const compressed = zlib.deflateRawSync(entry.data);
    const fileCrc = crc32(entry.data);

    const localHeader = Buffer.alloc(30);
    localHeader.writeUInt32LE(0x04034b50, 0);
    localHeader.writeUInt16LE(20, 4);
    localHeader.writeUInt16LE(0, 6);
    localHeader.writeUInt16LE(8, 8); // deflate
    localHeader.writeUInt16LE(0, 10);
    localHeader.writeUInt16LE(0, 12);
    localHeader.writeUInt32LE(fileCrc, 14);
    localHeader.writeUInt32LE(compressed.length, 18);
    localHeader.writeUInt32LE(entry.data.length, 22);
    localHeader.writeUInt16LE(nameBuf.length, 26);
    localHeader.writeUInt16LE(0, 28);

    localParts.push(localHeader, nameBuf, compressed);

    const centralHeader = Buffer.alloc(46);
    centralHeader.writeUInt32LE(0x02014b50, 0);
    centralHeader.writeUInt16LE(20, 4);
    centralHeader.writeUInt16LE(20, 6);
    centralHeader.writeUInt16LE(0, 8);
    centralHeader.writeUInt16LE(8, 10);
    centralHeader.writeUInt16LE(0, 12);
    centralHeader.writeUInt16LE(0, 14);
    centralHeader.writeUInt32LE(fileCrc, 16);
    centralHeader.writeUInt32LE(compressed.length, 20);
    centralHeader.writeUInt32LE(entry.data.length, 24);
    centralHeader.writeUInt16LE(nameBuf.length, 28);
    centralHeader.writeUInt16LE(0, 30);
    centralHeader.writeUInt16LE(0, 32);
    centralHeader.writeUInt16LE(0, 34);
    centralHeader.writeUInt16LE(0, 36);
    centralHeader.writeUInt32LE(0, 38);
    centralHeader.writeUInt32LE(offset, 42);

    centralParts.push(centralHeader, nameBuf);
    offset += localHeader.length + nameBuf.length + compressed.length;
  }

  const centralBuffer = Buffer.concat(centralParts);
  const endRecord = Buffer.alloc(22);
  endRecord.writeUInt32LE(0x06054b50, 0);
  endRecord.writeUInt16LE(0, 4);
  endRecord.writeUInt16LE(0, 6);
  endRecord.writeUInt16LE(entries.length, 8);
  endRecord.writeUInt16LE(entries.length, 10);
  endRecord.writeUInt32LE(centralBuffer.length, 12);
  endRecord.writeUInt32LE(offset, 16);
  endRecord.writeUInt16LE(0, 20);

  return Buffer.concat([...localParts, centralBuffer, endRecord]);
}

async function collectSafeSourceFiles(rootDir: string): Promise<Array<{ name: string; data: Buffer }>> {
  const entries: Array<{ name: string; data: Buffer }> = [];
  const rootFiles = [
    'package.json',
    'tsconfig.json',
    'vite.config.ts',
    'index.html',
    'metadata.json',
    '.env.example',
    '.gitignore',
    'capacitor.config.json',
    '.github/workflows/build-apk.yml',
    'server.ts',
  ];

  for (const rel of rootFiles) {
    const full = path.join(rootDir, rel);
    try {
      const data = await fs.readFile(full);
      entries.push({ name: `enhance-app/${rel}`, data });
    } catch {
      // Skip optional files
    }
  }

  async function walk(dir: string, prefix: string) {
    const items = await fs.readdir(dir, { withFileTypes: true });
    for (const item of items) {
      if (item.name.startsWith('.')) continue;
      const full = path.join(dir, item.name);
      const relName = `${prefix}/${item.name}`;
      if (item.isDirectory()) {
        await walk(full, relName);
      } else if (item.isFile()) {
        const data = await fs.readFile(full);
        entries.push({ name: `enhance-app/${relName}`, data });
      }
    }
  }

  for (const dirName of ['src', 'public']) {
    const targetDir = path.join(rootDir, dirName);
    try {
      await walk(targetDir, dirName);
    } catch {
      // Ignore if missing
    }
  }

  const readme = `# Enhance — Free AI App Blocker & Task Gate

## Run Locally
1. Install Node.js 22+ and run:
   npm install
2. Copy .env.example to .env and set GEMINI_API_KEY to your Gemini API key.
3. Run:
   npm run dev
4. Open http://localhost:3000 in your browser.

## Build an Android .apk
### Option A: Free Cloud APK Build (GitHub Actions)
1. Push this folder to a GitHub repository (make sure the \`.github\` folder is included).
2. Open the **Actions** tab and run **Build Android APK**.
3. Download the generated **enhance-android-apk** artifact (\`app-debug.apk\`).

### Option B: Local Android Studio / Capacitor
1. Ensure Java 17 and Android Studio / Android SDK are installed.
2. Run:
   npm install
   npm run build
   npm install @capacitor/core @capacitor/cli @capacitor/android
   npx cap add android
   npx cap sync android
   npx cap open android
3. In Android Studio, click Build -> Build Bundle(s) / APK(s) -> Build APK(s).
`;
  entries.push({ name: 'enhance-app/README.md', data: Buffer.from(readme, 'utf8') });

  return entries;
}

async function startServer() {
  const app = express();
  const httpServer = http.createServer(app);
  app.set('trust proxy', 1);
  app.use(express.json({ limit: '6mb' }));

  app.get('/api/health', (_req, res) => {
    res.json({ ok: true });
  });

  // Download clean source bundle as .zip
  const handleDownloadZip = async (_req: Request, res: Response) => {
    try {
      const files = await collectSafeSourceFiles(process.cwd());
      const zipBuf = buildZipBuffer(files);
      res.setHeader('Content-Type', 'application/zip');
      res.setHeader('Content-Disposition', 'attachment; filename="enhance-app.zip"');
      res.setHeader('Content-Length', String(zipBuf.length));
      res.send(zipBuf);
    } catch (err) {
      console.error('Download bundle error:', err);
      res.status(500).json({ error: 'Unable to package application bundle.' });
    }
  };
  app.get('/api/download/enhance-app.zip', handleDownloadZip);
  app.get('/api/download/critique-app.zip', handleDownloadZip);

  async function resolveApkLocation(): Promise<{ type: 'file'; path: string } | { type: 'url'; url: string } | null> {
    const localApk = path.resolve(process.cwd(), 'public', 'critique.apk');
    try {
      const stat = await fs.stat(localApk);
      if (stat.isFile() && stat.size > 0) {
        return { type: 'file', path: localApk };
      }
    } catch {
      // No local prebuilt APK yet
    }

    const envUrl = process.env.APK_DOWNLOAD_URL?.trim();
    if (envUrl) {
      try {
        const parsed = new URL(envUrl);
        if (parsed.protocol === 'https:') {
          return { type: 'url', url: parsed.toString() };
        }
      } catch {
        // Ignore invalid URL
      }
    }
    return null;
  }

  app.get('/api/download/apk-status', async (_req, res) => {
    const loc = await resolveApkLocation();
    res.json({
      available: loc !== null,
      downloadPath: loc ? '/api/download/critique.apk' : null,
    });
  });

  app.get('/api/download/critique.apk', async (_req, res) => {
    const loc = await resolveApkLocation();
    if (!loc) {
      res.status(404).json({
        error:
          'No compiled critique.apk has been published yet. Build once via GitHub Actions and place public/critique.apk or set APK_DOWNLOAD_URL.',
      });
      return;
    }
    if (loc.type === 'url') {
      res.redirect(302, loc.url);
      return;
    }
    res.setHeader('Content-Type', 'application/vnd.android.package-archive');
    res.setHeader('Content-Disposition', 'attachment; filename="critique.apk"');
    res.sendFile(loc.path);
  });

  app.post('/api/auth/register', async (req, res) => {
    try {
      const { email, password, name, age, mainGoal, selectedGoals, accent, critiquePersona, theme } =
        req.body ?? {};
      const emailCheck = validateEmailStrict(email);
      if (!emailCheck.valid) {
        res.status(400).json({ error: emailCheck.error });
        return;
      }

      const passCheck = validatePasswordStrength(password);
      if (!passCheck.valid) {
        res.status(400).json({ error: passCheck.error });
        return;
      }

      let parsedAge: number | null = null;
      if (age !== undefined && age !== null && age !== '') {
        const numericAge = Number(age);
        if (!Number.isInteger(numericAge) || numericAge < 10 || numericAge > 99) {
          res.status(400).json({ error: 'Please enter a valid age.' });
          return;
        }
        parsedAge = numericAge;
      }

      const validThemes = new Set(['light', 'dark']);
      const validAccents = new Set(['indigo', 'emerald', 'cobalt', 'amber']);
      const validPersonas = new Set(['supportive', 'balanced', 'strict']);

      const normalizedGoals = normalizeSelectedGoals(
        selectedGoals,
        VALID_GOALS.has(mainGoal) ? mainGoal : 'mindset'
      );

      const cleanName =
        typeof name === 'string' && name.trim().length >= 2
          ? name.trim().slice(0, 48)
          : emailCheck.normalized.split('@')[0];

      const db = await loadDb();
      if (db.users[emailCheck.normalized]) {
        res.status(409).json({ error: 'An account with this email already exists. Please sign in.' });
        return;
      }

      const { hash, salt } = hashPassword(password);
      const newUser: StoredUser = {
        id: crypto.randomUUID(),
        email: emailCheck.normalized,
        name: cleanName,
        passwordHash: hash,
        salt,
        createdAt: new Date().toISOString(),
        xp: 0,
        streakDays: 0,
        streakFreezes: 0,
        checkInsCompleted: 0,
        lastActiveDate: null,
        focusCoins: 25,
        totalMinutesLocked: 0,
        totalTasksPassed: 0,
        bestScore: 0,
        settings: {
          theme: validThemes.has(theme) ? theme : 'dark',
          accent: validAccents.has(accent) ? accent : 'indigo',
          critiquePersona: validPersonas.has(critiquePersona) ? critiquePersona : 'balanced',
          hardcoreMode: false,
          soundEnabled: true,
          pushNotificationsEnabled: false,
          density: 'comfortable',
          displayName: cleanName,
          age: parsedAge,
          mainGoal: normalizedGoals[0],
          selectedGoals: normalizedGoals,
        },
        customApps: [],
        activeLocks: [],
        routines: [],
        lastRoutineResetDate: todayKey(),
        rewardedRoutineStepKeys: [],
        critiqueHistory: [],
        unlockedAchievements: [],
        claimedQuestKeys: [],
      };

      db.users[emailCheck.normalized] = newUser;
      issueRememberCookie(req, res, newUser);
      await persistDb(db);

      const token = signSessionToken(newUser);
      res.status(201).json({
        token,
        user: sanitizeUser(newUser),
      });
    } catch (err) {
      console.error('Register error:', err);
      res.status(500).json({ error: 'Unable to create account right now.' });
    }
  });

  app.post('/api/auth/login', async (req, res) => {
    try {
      const { email, password } = req.body ?? {};
      const emailCheck = validateEmailStrict(email);
      if (!emailCheck.valid) {
        res.status(400).json({ error: emailCheck.error });
        return;
      }

      if (typeof password !== 'string' || password.length === 0) {
        res.status(400).json({ error: 'Please enter your password.' });
        return;
      }

      const rateKey = `${req.ip}:${emailCheck.normalized}`;
      const attempt = loginAttempts.get(rateKey);
      if (attempt && attempt.count >= 5 && Date.now() < attempt.resetAt) {
        const waitSeconds = Math.ceil((attempt.resetAt - Date.now()) / 1000);
        res.status(429).json({
          error: `Too many failed sign-in attempts. Wait ${waitSeconds}s before trying again.`,
        });
        return;
      }

      const db = await loadDb();
      const user = db.users[emailCheck.normalized];
      if (!user || !user.passwordHash || !verifyPassword(password, user.passwordHash, user.salt)) {
        const nextCount = attempt && Date.now() < attempt.resetAt ? attempt.count + 1 : 1;
        loginAttempts.set(rateKey, { count: nextCount, resetAt: Date.now() + 5 * 60 * 1000 });
        res.status(401).json({ error: 'Invalid email or password.' });
        return;
      }

      loginAttempts.delete(rateKey);
      syncExpiredLocks(user);
      syncDailyRoutineReset(user);
      issueRememberCookie(req, res, user);
      await persistDb(db);

      const token = signSessionToken(user);
      res.json({
        token,
        user: sanitizeUser(user),
      });
    } catch (err) {
      console.error('Login error:', err);
      res.status(500).json({ error: 'Unable to sign in right now.' });
    }
  });

  // Guest Account Creation & Instant Sign-In
  app.post('/api/auth/guest', async (req, res) => {
    try {
      const rateKey = `guest:${req.ip}`;
      const attempt = loginAttempts.get(rateKey);
      if (attempt && attempt.count >= 20 && Date.now() < attempt.resetAt) {
        const waitSeconds = Math.ceil((attempt.resetAt - Date.now()) / 1000);
        res.status(429).json({
          error: `Too many guest accounts created recently. Wait ${waitSeconds}s before trying again.`,
        });
        return;
      }
      const nextCount = attempt && Date.now() < attempt.resetAt ? attempt.count + 1 : 1;
      loginAttempts.set(rateKey, { count: nextCount, resetAt: Date.now() + 15 * 60 * 1000 });

      const { name, age, mainGoal, selectedGoals, accent, critiquePersona, theme } = req.body ?? {};

      let parsedAge: number | null = null;
      if (age !== undefined && age !== null && age !== '') {
        const numericAge = Number(age);
        if (!Number.isInteger(numericAge) || numericAge < 10 || numericAge > 99) {
          res.status(400).json({ error: 'Please enter a valid age.' });
          return;
        }
        parsedAge = numericAge;
      }

      const validThemes = new Set(['light', 'dark']);
      const validAccents = new Set(['indigo', 'emerald', 'cobalt', 'amber']);
      const validPersonas = new Set(['supportive', 'balanced', 'strict']);

      const normalizedGoals = normalizeSelectedGoals(
        selectedGoals,
        VALID_GOALS.has(mainGoal) ? mainGoal : 'mindset'
      );

      const guestSuffix = crypto.randomBytes(2).toString('hex').toUpperCase();
      const guestTag = crypto.randomUUID().slice(0, 8);
      const cleanName =
        typeof name === 'string' && name.trim().length >= 2
          ? name.trim().slice(0, 48)
          : `Guest ${guestSuffix}`;
      const guestEmail = `guest.${guestTag}@guest.enhance`;

      const db = await loadDb();
      const newUser: StoredUser = {
        id: crypto.randomUUID(),
        email: guestEmail,
        name: cleanName,
        authProvider: 'guest',
        passwordHash: '',
        salt: '',
        createdAt: new Date().toISOString(),
        xp: 0,
        streakDays: 0,
        streakFreezes: 0,
        checkInsCompleted: 0,
        lastActiveDate: null,
        focusCoins: 25,
        totalMinutesLocked: 0,
        totalTasksPassed: 0,
        bestScore: 0,
        settings: {
          theme: validThemes.has(theme) ? theme : 'dark',
          accent: validAccents.has(accent) ? accent : 'indigo',
          critiquePersona: validPersonas.has(critiquePersona) ? critiquePersona : 'balanced',
          hardcoreMode: false,
          soundEnabled: true,
          pushNotificationsEnabled: false,
          density: 'comfortable',
          displayName: cleanName,
          age: parsedAge,
          mainGoal: normalizedGoals[0],
          selectedGoals: normalizedGoals,
        },
        customApps: [],
        activeLocks: [],
        routines: [],
        lastRoutineResetDate: todayKey(),
        rewardedRoutineStepKeys: [],
        critiqueHistory: [],
        unlockedAchievements: [],
        claimedQuestKeys: [],
      };

      db.users[guestEmail] = newUser;
      issueRememberCookie(req, res, newUser);
      await persistDb(db);

      const token = signSessionToken(newUser);
      res.status(201).json({
        token,
        user: sanitizeUser(newUser),
      });
    } catch (err) {
      console.error('Guest sign-in error:', err);
      res.status(500).json({ error: 'Unable to create guest account right now.' });
    }
  });

  // Third-Party Social OAuth Discovery (Facebook, Instagram, TikTok)
  app.get('/api/auth/oauth/providers', (_req, res) => {
    const fb = getProviderConfig('facebook');
    const ig = getProviderConfig('instagram');
    const tt = getProviderConfig('tiktok');
    res.json({
      providers: {
        facebook: {
          configured: fb.configured,
          callbackUrl: fb.callbackUrl,
          requiredVars: fb.requiredVars,
        },
        instagram: {
          configured: ig.configured,
          callbackUrl: ig.callbackUrl,
          requiredVars: ig.requiredVars,
        },
        tiktok: {
          configured: tt.configured,
          callbackUrl: tt.callbackUrl,
          requiredVars: tt.requiredVars,
        },
      },
    });
  });

  // Build Official OAuth Authorization URL for Popup
  app.post('/api/auth/oauth/:provider/start', (req, res) => {
    const rawProvider = req.params.provider;
    if (rawProvider !== 'facebook' && rawProvider !== 'instagram' && rawProvider !== 'tiktok') {
      res.status(404).json({ error: 'Unsupported sign-in provider.' });
      return;
    }

    const provider: SocialOAuthProvider = rawProvider;
    const cfg = getProviderConfig(provider);
    if (!cfg.configured) {
      res.status(503).json({
        setupRequired: true,
        provider,
        callbackUrl: cfg.callbackUrl,
        requiredVars: cfg.requiredVars,
        error: `Add ${cfg.requiredVars.join(' and ')} in AI Studio Secrets to enable ${provider} sign-in.`,
      });
      return;
    }

    pruneOAuthMaps();
    const state = crypto.randomBytes(24).toString('hex');
    const body = req.body ?? {};
    const parsedAge =
      Number.isInteger(Number(body.age)) && Number(body.age) >= 10 && Number(body.age) <= 99
        ? Number(body.age)
        : undefined;
    const validThemes = new Set(['light', 'dark']);
    const validAccents = new Set(['indigo', 'emerald', 'cobalt', 'amber']);
    const validPersonas = new Set(['supportive', 'balanced', 'strict']);

    oauthStates.set(state, {
      provider,
      expiresAt: Date.now() + 10 * 60 * 1000,
      preAuth: {
        age: parsedAge,
        selectedGoals: normalizeSelectedGoals(body.selectedGoals, 'mindset'),
        theme: validThemes.has(body.theme) ? body.theme : 'dark',
        accent: validAccents.has(body.accent) ? body.accent : 'indigo',
        critiquePersona: validPersonas.has(body.critiquePersona)
          ? body.critiquePersona
          : 'balanced',
      },
    });

    let authUrl = '';
    if (provider === 'facebook') {
      const params = new URLSearchParams({
        client_id: cfg.clientId,
        redirect_uri: cfg.callbackUrl,
        state,
        response_type: 'code',
        scope: 'public_profile,email',
      });
      authUrl = `https://www.facebook.com/v21.0/dialog/oauth?${params.toString()}`;
    } else if (provider === 'instagram') {
      const params = new URLSearchParams({
        client_id: cfg.clientId,
        redirect_uri: cfg.callbackUrl,
        state,
        response_type: 'code',
        scope: 'instagram_business_basic',
      });
      authUrl = `https://www.instagram.com/oauth/authorize?${params.toString()}`;
    } else {
      const params = new URLSearchParams({
        client_key: cfg.clientId,
        redirect_uri: cfg.callbackUrl,
        state,
        response_type: 'code',
        scope: 'user.info.basic',
      });
      authUrl = `https://www.tiktok.com/v2/auth/authorize/?${params.toString()}`;
    }

    res.json({ url: authUrl });
  });

  // OAuth Callback Popup Responder (handles both with and without trailing slash)
  app.get(
    ['/auth/oauth/:provider/callback', '/auth/oauth/:provider/callback/'],
    async (req, res) => {
      const rawProvider = req.params.provider;
      if (rawProvider !== 'facebook' && rawProvider !== 'instagram' && rawProvider !== 'tiktok') {
        res.status(404).send('Unsupported OAuth provider.');
        return;
      }
      const provider: SocialOAuthProvider = rawProvider;

      const sendPopupResult = (payload: Record<string, unknown>, statusCode = 200) => {
        const serialized = JSON.stringify(payload).replace(/</g, '\\u003c');
        res.status(statusCode).setHeader('Content-Type', 'text/html; charset=utf-8').send(`<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Enhance Sign-In</title></head>
  <body style="background:#000;color:#fff;font-family:system-ui,sans-serif;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;">
    <div style="text-align:center;padding:24px;">
      <p style="font-size:14px;margin:0 0 8px;">${payload.type === 'OAUTH_AUTH_SUCCESS' ? 'Sign-in verified. Returning to Enhance...' : 'Authentication did not complete.'}</p>
    </div>
    <script>
      (function() {
        var msg = ${serialized};
        if (window.opener) {
          window.opener.postMessage(msg, '*');
          window.close();
        }
      })();
    </script>
  </body>
</html>`);
      };

      const { code, state, error: oauthError } = req.query;
      if (oauthError || typeof code !== 'string' || typeof state !== 'string') {
        sendPopupResult({ type: 'OAUTH_AUTH_ERROR', provider, error: 'Sign-in was cancelled or denied.' }, 400);
        return;
      }

      const pending = oauthStates.get(state);
      oauthStates.delete(state);
      if (!pending || pending.provider !== provider || Date.now() > pending.expiresAt) {
        sendPopupResult({ type: 'OAUTH_AUTH_ERROR', provider, error: 'Invalid or expired sign-in session. Please try again.' }, 400);
        return;
      }

      const cfg = getProviderConfig(provider);
      if (!cfg.configured) {
        sendPopupResult({ type: 'OAUTH_AUTH_ERROR', provider, error: 'OAuth provider is not configured.' }, 503);
        return;
      }

      try {
        let providerId = '';
        let displayName = '';
        let verifiedEmail: string | null = null;

        if (provider === 'facebook') {
          const tokenParams = new URLSearchParams({
            client_id: cfg.clientId,
            client_secret: cfg.clientSecret,
            redirect_uri: cfg.callbackUrl,
            code,
          });
          const tokenRes = await fetch(
            `https://graph.facebook.com/v21.0/oauth/access_token?${tokenParams.toString()}`
          );
          const tokenJson = (await tokenRes.json()) as { access_token?: string };
          if (!tokenRes.ok || !tokenJson.access_token) {
            throw new Error('Facebook token exchange failed.');
          }
          const profileRes = await fetch(
            `https://graph.facebook.com/v21.0/me?fields=id,name,email&access_token=${encodeURIComponent(tokenJson.access_token)}`
          );
          const profile = (await profileRes.json()) as { id?: string; name?: string; email?: string };
          if (!profileRes.ok || !profile.id) {
            throw new Error('Unable to read Facebook profile.');
          }
          providerId = profile.id;
          displayName = profile.name?.trim() || 'Facebook Member';
          if (profile.email && validateEmailStrict(profile.email).valid) {
            verifiedEmail = validateEmailStrict(profile.email).normalized;
          }
        } else if (provider === 'instagram') {
          const form = new URLSearchParams({
            client_id: cfg.clientId,
            client_secret: cfg.clientSecret,
            grant_type: 'authorization_code',
            redirect_uri: cfg.callbackUrl,
            code,
          });
          const tokenRes = await fetch('https://api.instagram.com/oauth/access_token', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: form.toString(),
          });
          const tokenJson = (await tokenRes.json()) as {
            access_token?: string;
            user_id?: string | number;
          };
          if (!tokenRes.ok || !tokenJson.access_token) {
            throw new Error('Instagram token exchange failed.');
          }
          const profileRes = await fetch(
            `https://graph.instagram.com/v21.0/me?fields=user_id,username,name&access_token=${encodeURIComponent(tokenJson.access_token)}`
          );
          const profile = (await profileRes.json()) as {
            user_id?: string;
            id?: string;
            username?: string;
            name?: string;
          };
          providerId = String(profile.user_id || profile.id || tokenJson.user_id || '');
          if (!providerId) {
            throw new Error('Unable to read Instagram profile.');
          }
          displayName = profile.name?.trim() || (profile.username ? `@${profile.username}` : 'Instagram Member');
        } else {
          const form = new URLSearchParams({
            client_key: cfg.clientId,
            client_secret: cfg.clientSecret,
            code,
            grant_type: 'authorization_code',
            redirect_uri: cfg.callbackUrl,
          });
          const tokenRes = await fetch('https://open.tiktokapis.com/v2/oauth/token/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: form.toString(),
          });
          const tokenJson = (await tokenRes.json()) as { access_token?: string; open_id?: string };
          if (!tokenRes.ok || !tokenJson.access_token) {
            throw new Error('TikTok token exchange failed.');
          }
          const profileRes = await fetch(
            'https://open.tiktokapis.com/v2/user/info/?fields=open_id,display_name',
            {
              headers: { Authorization: `Bearer ${tokenJson.access_token}` },
            }
          );
          const profileJson = (await profileRes.json()) as {
            data?: { user?: { open_id?: string; display_name?: string } };
          };
          const ttUser = profileJson.data?.user;
          providerId = ttUser?.open_id || tokenJson.open_id || '';
          if (!providerId) {
            throw new Error('Unable to read TikTok profile.');
          }
          displayName = ttUser?.display_name?.trim() || 'TikTok Member';
        }

        const db = await loadDb();
        let existingUser = Object.values(db.users).find(
          (u) => u.authProvider === provider && u.oauthProviderId === providerId
        );
        if (!existingUser && verifiedEmail) {
          existingUser = db.users[verifiedEmail];
        }

        if (!existingUser) {
          const accountKey = verifiedEmail || `oauth.${provider}.${providerId}@social.critique`;
          const goals = pending.preAuth?.selectedGoals?.length
            ? pending.preAuth.selectedGoals
            : (['mindset'] as SelfImprovementGoal[]);
          existingUser = {
            id: crypto.randomUUID(),
            email: accountKey,
            name: displayName.slice(0, 48),
            authProvider: provider,
            oauthProviderId: providerId,
            passwordHash: '',
            salt: '',
            createdAt: new Date().toISOString(),
            xp: 0,
            streakDays: 0,
            streakFreezes: 0,
            checkInsCompleted: 0,
            lastActiveDate: null,
            focusCoins: 25,
            totalMinutesLocked: 0,
            totalTasksPassed: 0,
            bestScore: 0,
            settings: {
              theme: pending.preAuth?.theme ?? 'dark',
              accent: pending.preAuth?.accent ?? 'indigo',
              critiquePersona: pending.preAuth?.critiquePersona ?? 'balanced',
              hardcoreMode: false,
              soundEnabled: true,
              pushNotificationsEnabled: false,
              density: 'comfortable',
              displayName: displayName.slice(0, 48),
              age: pending.preAuth?.age ?? null,
              mainGoal: goals[0],
              selectedGoals: goals,
            },
            customApps: [],
            activeLocks: [],
            routines: [],
            lastRoutineResetDate: todayKey(),
            rewardedRoutineStepKeys: [],
            critiqueHistory: [],
            unlockedAchievements: [],
            claimedQuestKeys: [],
          };
          db.users[accountKey] = existingUser;
          await persistDb(db);
        }

        const ticket = crypto.randomBytes(24).toString('hex');
        oauthTickets.set(ticket, {
          email: existingUser.email,
          provider,
          expiresAt: Date.now() + 60 * 1000,
        });

        sendPopupResult({ type: 'OAUTH_AUTH_SUCCESS', ticket, provider });
      } catch (err) {
        console.error(`OAuth ${provider} callback error:`, err);
        sendPopupResult(
          { type: 'OAUTH_AUTH_ERROR', provider, error: 'Could not complete social sign-in. Please try again.' },
          500
        );
      }
    }
  );

  // Exchange Short-Lived OAuth Ticket for Bearer Session Token
  app.post('/api/auth/oauth/exchange', async (req, res) => {
    const ticket = typeof req.body?.ticket === 'string' ? req.body.ticket.trim() : '';
    if (!ticket) {
      res.status(400).json({ error: 'Missing sign-in ticket.' });
      return;
    }

    const entry = oauthTickets.get(ticket);
    oauthTickets.delete(ticket);
    if (!entry || Date.now() > entry.expiresAt) {
      res.status(401).json({ error: 'Sign-in ticket expired. Please try again.' });
      return;
    }

    const db = await loadDb();
    const user = db.users[entry.email];
    if (!user) {
      res.status(404).json({ error: 'Account not found.' });
      return;
    }

    syncExpiredLocks(user);
    syncDailyRoutineReset(user);
    issueRememberCookie(req, res, user);
    await persistDb(db);

    const token = signSessionToken(user);
    res.json({
      token,
      user: sanitizeUser(user),
    });
  });

  app.post('/api/auth/change-password', requireAuth, async (req: AuthedRequest, res) => {
    const db = await loadDb();
    const user = db.users[req.authPayload!.email];
    if (!user) {
      res.status(404).json({ error: 'Account not found.' });
      return;
    }

    const { currentPassword, newPassword } = req.body ?? {};
    if (user.passwordHash) {
      if (
        typeof currentPassword !== 'string' ||
        !verifyPassword(currentPassword, user.passwordHash, user.salt)
      ) {
        res.status(401).json({ error: 'Current password is incorrect.' });
        return;
      }
    }

    const passCheck = validatePasswordStrength(newPassword);
    if (!passCheck.valid) {
      res.status(400).json({ error: passCheck.error });
      return;
    }

    const { hash, salt } = hashPassword(newPassword);
    user.passwordHash = hash;
    user.salt = salt;
    await persistDb(db);
    res.json({ ok: true });
  });

  app.get('/api/auth/me', requireAuth, async (req: AuthedRequest, res) => {
    const db = await loadDb();
    const user = db.users[req.authPayload!.email];
    if (!user) {
      res.status(404).json({ error: 'Account not found.' });
      return;
    }
    const locksChanged = syncExpiredLocks(user);
    const routinesReset = syncDailyRoutineReset(user);
    if (locksChanged || routinesReset) {
      await persistDb(db);
    }
    res.json({ user: sanitizeUser(user) });
  });

  // Restore remembered user session after leaving/exiting app via back key or reload
  app.get('/api/auth/session', async (req, res) => {
    try {
      const origin = req.get('origin');
      const expectedHost = (req.get('x-forwarded-host') || req.get('host') || '')
        .split(',')[0]
        .trim();
      if (origin && expectedHost) {
        try {
          if (new URL(origin).host !== expectedHost) {
            res.status(403).json({ error: 'Cross-origin session check denied.' });
            return;
          }
        } catch {
          res.status(403).json({ error: 'Invalid origin header.' });
          return;
        }
      }

      const cookies = parseCookies(req.headers.cookie);
      const rawRemember = cookies[REMEMBER_COOKIE_NAME];
      if (!rawRemember || rawRemember.length < 24) {
        res.status(401).json({ error: 'No remembered session found.' });
        return;
      }

      const incomingHash = hashRememberToken(rawRemember);
      const incomingBuf = Buffer.from(incomingHash, 'utf8');
      const now = Date.now();
      const db = await loadDb();

      let matchedUser: StoredUser | null = null;
      let dbModified = false;

      for (const candidate of Object.values(db.users)) {
        if (!Array.isArray(candidate.rememberTokens) || candidate.rememberTokens.length === 0) {
          continue;
        }
        for (const entry of candidate.rememberTokens) {
          if (entry.expiresAt <= now) continue;
          const entryBuf = Buffer.from(entry.hash, 'utf8');
          if (
            entryBuf.length === incomingBuf.length &&
            crypto.timingSafeEqual(entryBuf, incomingBuf)
          ) {
            matchedUser = candidate;
            break;
          }
        }
        if (matchedUser) break;
      }

      if (!matchedUser) {
        clearRememberCookie(req, res);
        res.status(401).json({ error: 'Session expired. Please sign in.' });
        return;
      }

      const beforeLen = matchedUser.rememberTokens?.length ?? 0;
      matchedUser.rememberTokens = (matchedUser.rememberTokens ?? []).filter(
        (entry) => entry.expiresAt > now
      );
      if (matchedUser.rememberTokens.length !== beforeLen) {
        dbModified = true;
      }

      if (syncExpiredLocks(matchedUser)) dbModified = true;
      if (syncDailyRoutineReset(matchedUser)) dbModified = true;
      if (dbModified) {
        await persistDb(db);
      }

      const token = signSessionToken(matchedUser);
      res.json({
        token,
        user: sanitizeUser(matchedUser),
      });
    } catch (err) {
      console.error('Session restore error:', err);
      res.status(500).json({ error: 'Unable to restore session.' });
    }
  });

  app.post('/api/auth/logout', requireAuth, async (req: AuthedRequest, res) => {
    if (req.authPayload?.jti) {
      revokedTokens.add(req.authPayload.jti);
    }
    const cookies = parseCookies(req.headers.cookie);
    const rawRemember = cookies[REMEMBER_COOKIE_NAME];
    if (rawRemember && req.authPayload?.email) {
      const incomingHash = hashRememberToken(rawRemember);
      const db = await loadDb();
      const user = db.users[req.authPayload.email];
      if (user && Array.isArray(user.rememberTokens)) {
        user.rememberTokens = user.rememberTokens.filter((entry) => entry.hash !== incomingHash);
        await persistDb(db);
      }
    }
    clearRememberCookie(req, res);
    res.json({ ok: true });
  });

  app.put('/api/user/settings', requireAuth, async (req: AuthedRequest, res) => {
    const db = await loadDb();
    const user = db.users[req.authPayload!.email];
    if (!user) {
      res.status(404).json({ error: 'Account not found.' });
      return;
    }

    const incoming = req.body ?? {};
    const validThemes = new Set(['light', 'dark']);
    const validAccents = new Set(['indigo', 'emerald', 'cobalt', 'amber']);
    const validPersonas = new Set(['supportive', 'balanced', 'strict']);
    const validDensities = new Set(['comfortable', 'compact']);
    const validGoals = new Set<SelfImprovementGoal>([
      'fitness',
      'looks_style',
      'social_skills',
      'mindset',
    ]);

    if (validThemes.has(incoming.theme)) user.settings.theme = incoming.theme;
    if (validAccents.has(incoming.accent)) user.settings.accent = incoming.accent;
    if (validPersonas.has(incoming.critiquePersona)) {
      user.settings.critiquePersona = incoming.critiquePersona;
    }
    if (typeof incoming.hardcoreMode === 'boolean') {
      user.settings.hardcoreMode = incoming.hardcoreMode;
    }
    if (typeof incoming.soundEnabled === 'boolean') {
      user.settings.soundEnabled = incoming.soundEnabled;
    }
    if (typeof incoming.pushNotificationsEnabled === 'boolean') {
      user.settings.pushNotificationsEnabled = incoming.pushNotificationsEnabled;
    }
    if (validDensities.has(incoming.density)) {
      user.settings.density = incoming.density;
    }
    if (Array.isArray(incoming.selectedGoals)) {
      const normalized = normalizeSelectedGoals(incoming.selectedGoals, user.settings.mainGoal);
      user.settings.selectedGoals = normalized;
      user.settings.mainGoal = normalized[0];
    } else if (validGoals.has(incoming.mainGoal)) {
      user.settings.mainGoal = incoming.mainGoal;
      user.settings.selectedGoals = normalizeSelectedGoals(
        [incoming.mainGoal, ...(user.settings.selectedGoals ?? []).filter((g) => g !== incoming.mainGoal)],
        incoming.mainGoal
      );
    }
    if (incoming.age !== undefined && incoming.age !== null && incoming.age !== '') {
      const numericAge = Number(incoming.age);
      if (!Number.isInteger(numericAge) || numericAge < 10 || numericAge > 99) {
        res.status(400).json({ error: 'Please enter a valid age.' });
        return;
      }
      user.settings.age = numericAge;
    }
    if (typeof incoming.displayName === 'string' && incoming.displayName.trim().length >= 2) {
      const clean = incoming.displayName.trim().slice(0, 48);
      user.settings.displayName = clean;
      user.name = clean;
    }

    await persistDb(db);
    res.json({ user: sanitizeUser(user) });
  });

  app.post('/api/apps/custom', requireAuth, async (req: AuthedRequest, res) => {
    const db = await loadDb();
    const user = db.users[req.authPayload!.email];
    if (!user) {
      res.status(404).json({ error: 'Account not found.' });
      return;
    }

    const { name, handle } = req.body ?? {};
    if (typeof name !== 'string' || name.trim().length < 2 || name.trim().length > 36) {
      res.status(400).json({ error: 'App name must be between 2 and 36 characters.' });
      return;
    }
    if (typeof handle !== 'string' || handle.trim().length < 2 || handle.trim().length > 60) {
      res.status(400).json({ error: 'App domain or handle must be between 2 and 60 characters.' });
      return;
    }

    const newApp: AppCatalogItem = {
      id: `custom-${crypto.randomUUID().slice(0, 8)}`,
      name: name.trim(),
      category: 'Custom',
      handle: handle.trim().toLowerCase().replace(/^https?:\/\//, ''),
      isCustom: true,
    };

    user.customApps.push(newApp);
    await persistDb(db);
    res.status(201).json({ app: newApp, user: sanitizeUser(user) });
  });

  app.post('/api/locks', requireAuth, async (req: AuthedRequest, res) => {
    const db = await loadDb();
    const user = db.users[req.authPayload!.email];
    if (!user) {
      res.status(404).json({ error: 'Account not found.' });
      return;
    }

    syncExpiredLocks(user);

    const { appIds, durationMinutes, taskTitle, taskCategory, taskPrompt } = req.body ?? {};

    if (!Array.isArray(appIds) || appIds.length === 0) {
      res.status(400).json({ error: 'Select at least one app to lock.' });
      return;
    }

    const duration = Number(durationMinutes);
    if (!Number.isFinite(duration) || duration < 5 || duration > 480) {
      res.status(400).json({ error: 'Lock duration must be between 5 and 480 minutes.' });
      return;
    }

    if (typeof taskTitle !== 'string' || taskTitle.trim().length < 3 || taskTitle.trim().length > 80) {
      res.status(400).json({ error: 'Task title must be between 3 and 80 characters.' });
      return;
    }

    if (
      typeof taskPrompt !== 'string' ||
      taskPrompt.trim().length < 10 ||
      taskPrompt.trim().length > 500
    ) {
      res.status(400).json({ error: 'Unlock task instructions must be 10 to 500 characters.' });
      return;
    }

    const allApps = [...DEFAULT_APPS, ...user.customApps];
    const now = Date.now();
    const unlocksAt = new Date(now + duration * 60 * 1000).toISOString();
    const lockedAt = new Date(now).toISOString();

    const createdLocks: ActiveLock[] = [];
    for (const appId of appIds) {
      const appItem = allApps.find((a) => a.id === appId);
      if (!appItem) continue;

      const alreadyLocked = user.activeLocks.some(
        (l) => l.appId === appItem.id && l.status === 'locked'
      );
      if (alreadyLocked) continue;

      const lock: ActiveLock = {
        id: `lock-${crypto.randomUUID().slice(0, 8)}`,
        appId: appItem.id,
        appName: appItem.name,
        appCategory: appItem.category,
        appHandle: appItem.handle,
        durationMinutes: Math.round(duration),
        lockedAt,
        unlocksAt,
        taskTitle: taskTitle.trim(),
        taskCategory:
          typeof taskCategory === 'string' && taskCategory.trim() ? taskCategory.trim() : 'Custom',
        taskPrompt: taskPrompt.trim(),
        status: 'locked',
        attemptsCount: 0,
      };
      user.activeLocks.unshift(lock);
      createdLocks.push(lock);
    }

    if (createdLocks.length === 0) {
      res.status(400).json({ error: 'Selected apps are already locked.' });
      return;
    }

    touchStreak(user);
    awardAchievement(user, 'first-lock');
    if (duration >= 60) awardAchievement(user, 'marathon-lock');
    const currentlyLockedCount = user.activeLocks.filter((l) => l.status === 'locked').length;
    if (currentlyLockedCount >= 3) awardAchievement(user, 'multi-vault');

    computeDailyQuests(user);
    await persistDb(db);

    res.status(201).json({ locks: createdLocks, user: sanitizeUser(user) });
  });

  app.post('/api/locks/claim-expired', requireAuth, async (req: AuthedRequest, res) => {
    const db = await loadDb();
    const user = db.users[req.authPayload!.email];
    if (!user) {
      res.status(404).json({ error: 'Account not found.' });
      return;
    }

    const changed = syncExpiredLocks(user);
    if (changed) {
      computeDailyQuests(user);
      await persistDb(db);
    }
    res.json({ changed, user: sanitizeUser(user) });
  });

  app.post('/api/locks/:id/cancel', requireAuth, async (req: AuthedRequest, res) => {
    const db = await loadDb();
    const user = db.users[req.authPayload!.email];
    if (!user) {
      res.status(404).json({ error: 'Account not found.' });
      return;
    }

    if (user.settings.hardcoreMode) {
      res.status(403).json({
        error:
          'Hardcore Lock Mode is active. Complete the unlock task or wait for the timer to expire.',
      });
      return;
    }

    const lock = user.activeLocks.find((l) => l.id === req.params.id && l.status === 'locked');
    if (!lock) {
      res.status(404).json({ error: 'Active lock not found.' });
      return;
    }

    lock.status = 'cancelled';
    await persistDb(db);
    res.json({ user: sanitizeUser(user) });
  });

  // Daily Self-Improvement Check-In Completion
  app.post('/api/gamification/checkin', requireAuth, async (req: AuthedRequest, res) => {
    const db = await loadDb();
    const user = db.users[req.authPayload!.email];
    if (!user) {
      res.status(404).json({ error: 'Account not found.' });
      return;
    }

    const validGoals = new Set<SelfImprovementGoal>([
      'fitness',
      'looks_style',
      'social_skills',
      'mindset',
    ]);
    const goal: SelfImprovementGoal = validGoals.has(req.body?.goal)
      ? req.body.goal
      : user.settings.mainGoal || 'mindset';
    const note = typeof req.body?.note === 'string' ? req.body.note.trim() : '';
    if (note.length < 5) {
      res.status(400).json({ error: 'Write a quick summary (at least 5 characters) of the action you completed.' });
      return;
    }

    const goalMeta = SELF_IMPROVEMENT_GOALS.find((g) => g.id === goal);
    user.checkInsCompleted = (user.checkInsCompleted ?? 0) + 1;
    user.xp += 40;
    user.focusCoins += 15;
    touchStreak(user);
    computeDailyQuests(user);
    await persistDb(db);

    res.json({
      xpEarned: 40,
      coinsEarned: 15,
      goalLabel: goalMeta?.label || 'Mindset',
      user: sanitizeUser(user),
    });
  });

  // Create Custom Routine
  app.post('/api/routines', requireAuth, async (req: AuthedRequest, res) => {
    const db = await loadDb();
    const user = db.users[req.authPayload!.email];
    if (!user) {
      res.status(404).json({ error: 'Account not found.' });
      return;
    }
    syncDailyRoutineReset(user);

    const { title, focusGoal, focusGoals, steps } = req.body ?? {};
    if (typeof title !== 'string' || title.trim().length < 2 || title.trim().length > 80) {
      res.status(400).json({ error: 'Routine title must be between 2 and 80 characters.' });
      return;
    }
    if (!Array.isArray(steps) || steps.length === 0 || steps.length > 12) {
      res.status(400).json({ error: 'Include between 1 and 12 routine steps.' });
      return;
    }

    const validCategories = new Set<RoutineStep['category']>([
      'Fitness',
      'Looks & Style',
      'Social Skills',
      'Mindset',
    ]);

    const cleanSteps: RoutineStep[] = [];
    for (const raw of steps) {
      const stepTitle = typeof raw?.title === 'string' ? raw.title.trim() : '';
      const timeSlot =
        typeof raw?.timeSlot === 'string' && raw.timeSlot.trim()
          ? raw.timeSlot.trim().slice(0, 32)
          : 'Anytime';
      const category: RoutineStep['category'] = validCategories.has(raw?.category)
        ? raw.category
        : 'Mindset';
      const howToDo =
        typeof raw?.howToDo === 'string' && raw.howToDo.trim()
          ? raw.howToDo.trim().slice(0, 360)
          : undefined;
      const whyImportant =
        typeof raw?.whyImportant === 'string' && raw.whyImportant.trim()
          ? raw.whyImportant.trim().slice(0, 280)
          : undefined;
      if (stepTitle.length < 2) continue;
      cleanSteps.push({
        id: `step-${crypto.randomUUID().slice(0, 8)}`,
        title: stepTitle.slice(0, 140),
        timeSlot,
        category,
        ...(howToDo ? { howToDo } : {}),
        ...(whyImportant ? { whyImportant } : {}),
        completedToday: false,
      });
    }

    if (cleanSteps.length === 0) {
      res.status(400).json({ error: 'Each routine step needs a clear description.' });
      return;
    }

    const normalizedRoutineGoals = normalizeSelectedGoals(
      focusGoals,
      VALID_GOALS.has(focusGoal) ? focusGoal : user.settings.mainGoal || 'mindset'
    );

    const routine: UserRoutine = {
      id: `routine-${crypto.randomUUID().slice(0, 8)}`,
      title: title.trim(),
      source: 'custom',
      focusGoal: normalizedRoutineGoals[0],
      focusGoals: normalizedRoutineGoals,
      createdAt: new Date().toISOString(),
      steps: cleanSteps,
    };

    user.routines = [routine, ...(user.routines ?? [])];
    await persistDb(db);
    res.status(201).json({ routine, user: sanitizeUser(user) });
  });

  // AI Personalized Routine Generator (age-aware, multi-skill, with How To + Why Important per task)
  app.post('/api/routines/generate', requireAuth, async (req: AuthedRequest, res) => {
    try {
      const db = await loadDb();
      const user = db.users[req.authPayload!.email];
      if (!user) {
        res.status(404).json({ error: 'Account not found.' });
        return;
      }
      syncDailyRoutineReset(user);

      const { focusGoal, focusGoals, age, intensity, scheduleNote } = req.body ?? {};
      let effectiveAge = user.settings.age;
      if (age !== undefined && age !== null && age !== '') {
        const numericAge = Number(age);
        if (!Number.isInteger(numericAge) || numericAge < 10 || numericAge > 99) {
          res.status(400).json({ error: 'Please enter a valid age between 10 and 99.' });
          return;
        }
        effectiveAge = numericAge;
        user.settings.age = numericAge;
      }

      const fallbackGoal = VALID_GOALS.has(focusGoal)
        ? (focusGoal as SelfImprovementGoal)
        : user.settings.mainGoal || 'mindset';
      const chosenGoals = normalizeSelectedGoals(
        Array.isArray(focusGoals) && focusGoals.length > 0
          ? focusGoals
          : user.settings.selectedGoals,
        fallbackGoal
      );

      const chosenGoalMetas = chosenGoals.map(
        (id) => SELF_IMPROVEMENT_GOALS.find((g) => g.id === id) ?? SELF_IMPROVEMENT_GOALS[3]
      );
      const goalSummaryText = chosenGoalMetas
        .map((g) => `${g.label} (${g.description})`)
        .join('; ');
      const chosenIntensity =
        intensity === 'light' || intensity === 'intense' ? intensity : 'balanced';
      const cleanNote = typeof scheduleNote === 'string' ? scheduleNote.trim().slice(0, 240) : '';
      const ageContext = effectiveAge
        ? `User Age: ${effectiveAge} years old. Tailor every exercise volume, social challenge, grooming step, and mindset practice specifically to be safe, realistic, and impactful for a ${effectiveAge}-year-old.`
        : 'User Age: Not specified (keep tasks universally safe, practical, and adaptable).';
      const levelInfo = computeLevelInfo(user.xp);
      const targetStepCount = chosenGoals.length >= 3 ? Math.min(8, chosenGoals.length * 2) : 5;

      const ai = createGeminiClient();
      const requestPayload = {
        model: 'gemini-4-hydrogen',
        contents: `Design a complete, age-appropriate daily self-improvement routine.
Selected Skills to Cover: ${goalSummaryText}
${ageContext}
Current Rank: ${levelInfo.rankTitle} (Level ${levelInfo.level})
Intensity: ${chosenIntensity}
User Schedule / Preferences: ${cleanNote || 'Standard daily schedule'}
Requirements:
- Return ${targetStepCount} concrete, actionable steps timed across the day (Morning, Midday, Afternoon, Evening) covering ALL selected skills (${chosenGoalMetas.map((g) => g.label).join(', ')}).
- Keep every step strictly centered on Fitness, Looks & Style, Social Skills, and Mindset (no school, homework, study, or coding tasks).
- Set "category" for each step to one of: "Fitness", "Looks & Style", "Social Skills", or "Mindset".
- For EVERY single step, provide:
  1. "howToDo": Specific, clear step-by-step instructions (2-3 sentences) telling the user exactly HOW to perform the task properly.
  2. "whyImportant": Clear, motivating explanation (1-2 sentences) telling the user WHY this task matters for their growth and age.`,
        config: {
          systemInstruction:
            'You are Critique AI, a practical self-improvement coach specializing in Fitness, Looks & Style, Social Skills, and Mindset. You design realistic age-tailored daily routines and always explain clearly how to perform each task and why it matters.',
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              title: { type: Type.STRING },
              steps: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    timeSlot: { type: Type.STRING },
                    category: { type: Type.STRING },
                    title: { type: Type.STRING },
                    howToDo: { type: Type.STRING },
                    whyImportant: { type: Type.STRING },
                  },
                  required: ['timeSlot', 'category', 'title', 'howToDo', 'whyImportant'],
                },
              },
            },
            required: ['title', 'steps'],
          },
        },
      };

      let aiResponse;
      try {
        aiResponse = await ai.models.generateContent(requestPayload);
      } catch (firstErr: unknown) {
        const firstMsg = firstErr instanceof Error ? firstErr.message : String(firstErr);
        if (firstMsg.includes('503') || firstMsg.includes('UNAVAILABLE')) {
          await new Promise((r) => setTimeout(r, 1500));
          aiResponse = await ai.models.generateContent(requestPayload);
        } else {
          throw firstErr;
        }
      }

      const parsed = JSON.parse((aiResponse.text ?? '{}').trim()) as {
        title?: string;
        steps?: Array<{
          timeSlot?: string;
          category?: string;
          title?: string;
          howToDo?: string;
          whyImportant?: string;
        }>;
      };

      const validCategories = new Set<RoutineStep['category']>([
        'Fitness',
        'Looks & Style',
        'Social Skills',
        'Mindset',
      ]);

      const cleanSteps: RoutineStep[] = (Array.isArray(parsed.steps) ? parsed.steps : [])
        .slice(0, 8)
        .map((s) => ({
          id: `step-${crypto.randomUUID().slice(0, 8)}`,
          timeSlot:
            typeof s.timeSlot === 'string' && s.timeSlot.trim()
              ? s.timeSlot.trim().slice(0, 32)
              : 'Daily',
          category: validCategories.has(s.category as RoutineStep['category'])
            ? (s.category as RoutineStep['category'])
            : (chosenGoalMetas[0].label as RoutineStep['category']),
          title:
            typeof s.title === 'string' && s.title.trim()
              ? s.title.trim().slice(0, 140)
              : 'Complete daily self-improvement rep',
          howToDo:
            typeof s.howToDo === 'string' && s.howToDo.trim()
              ? s.howToDo.trim().slice(0, 360)
              : 'Perform this habit with full focus for 10-15 minutes without checking distracting apps.',
          whyImportant:
            typeof s.whyImportant === 'string' && s.whyImportant.trim()
              ? s.whyImportant.trim().slice(0, 280)
              : 'Builds daily consistency, confidence, and measurable momentum in your core pillars.',
          completedToday: false,
        }));

      if (cleanSteps.length === 0) {
        res.status(502).json({ error: 'AI routine response was empty. Please try generating again.' });
        return;
      }

      const defaultTitle =
        chosenGoalMetas.length === SELF_IMPROVEMENT_GOALS.length
          ? 'All-Around Self-Mastery Routine'
          : `${chosenGoalMetas.map((g) => g.label).join(' + ')} Routine`;

      const routine: UserRoutine = {
        id: `routine-${crypto.randomUUID().slice(0, 8)}`,
        title:
          typeof parsed.title === 'string' && parsed.title.trim()
            ? parsed.title.trim().slice(0, 80)
            : defaultTitle.slice(0, 80),
        source: 'ai',
        focusGoal: chosenGoals[0],
        focusGoals: chosenGoals,
        createdAt: new Date().toISOString(),
        steps: cleanSteps,
      };

      user.routines = [routine, ...(user.routines ?? [])];
      await persistDb(db);
      res.status(201).json({ routine, user: sanitizeUser(user) });
    } catch (err: unknown) {
      console.error('Generate routine error:', err);
      const message = err instanceof Error ? err.message : 'Unable to generate routine right now.';
      if (message.includes('GEMINI_API_KEY')) {
        res.status(503).json({
          error: 'Gemini API key is not configured yet. Add GEMINI_API_KEY in Settings > Secrets.',
        });
        return;
      }
      if (message.includes('503') || message.includes('UNAVAILABLE')) {
        res.status(503).json({
          error: 'The AI routine designer is experiencing temporary high demand. Please tap Design My Personalized Routine again in a few seconds.',
        });
        return;
      }
      res.status(500).json({ error: `AI routine generation failed: ${message}` });
    }
  });

  // Toggle Routine Step Completion (+15 XP & +5 Coins once per step per day)
  app.post('/api/routines/:routineId/steps/:stepId/toggle', requireAuth, async (req: AuthedRequest, res) => {
    const db = await loadDb();
    const user = db.users[req.authPayload!.email];
    if (!user) {
      res.status(404).json({ error: 'Account not found.' });
      return;
    }
    syncDailyRoutineReset(user);

    const routine = (user.routines ?? []).find((r) => r.id === req.params.routineId);
    if (!routine) {
      res.status(404).json({ error: 'Routine not found.' });
      return;
    }

    const step = routine.steps.find((s) => s.id === req.params.stepId);
    if (!step) {
      res.status(404).json({ error: 'Routine step not found.' });
      return;
    }

    step.completedToday = !step.completedToday;
    let xpEarned = 0;
    let coinsEarned = 0;

    if (step.completedToday) {
      const rewardKey = `${todayKey()}:${step.id}`;
      user.rewardedRoutineStepKeys = user.rewardedRoutineStepKeys ?? [];
      if (!user.rewardedRoutineStepKeys.includes(rewardKey)) {
        user.rewardedRoutineStepKeys.push(rewardKey);
        xpEarned = 15;
        coinsEarned = 5;
        user.xp += xpEarned;
        user.focusCoins += coinsEarned;
      }
      touchStreak(user);
    }

    computeDailyQuests(user);
    await persistDb(db);
    res.json({ completedToday: step.completedToday, xpEarned, coinsEarned, user: sanitizeUser(user) });
  });

  // Delete Routine
  app.delete('/api/routines/:routineId', requireAuth, async (req: AuthedRequest, res) => {
    const db = await loadDb();
    const user = db.users[req.authPayload!.email];
    if (!user) {
      res.status(404).json({ error: 'Account not found.' });
      return;
    }
    const before = (user.routines ?? []).length;
    user.routines = (user.routines ?? []).filter((r) => r.id !== req.params.routineId);
    if (user.routines.length === before) {
      res.status(404).json({ error: 'Routine not found.' });
      return;
    }
    await persistDb(db);
    res.json({ user: sanitizeUser(user) });
  });

  // Focus Coin Rewards Shop
  app.post('/api/gamification/shop/buy', requireAuth, async (req: AuthedRequest, res) => {
    const db = await loadDb();
    const user = db.users[req.authPayload!.email];
    if (!user) {
      res.status(404).json({ error: 'Account not found.' });
      return;
    }

    const item = req.body?.item;
    if (item === 'streak_freeze') {
      if (user.focusCoins < 50) {
        res.status(400).json({ error: 'You need 50 Focus Coins for a Streak Freeze Shield.' });
        return;
      }
      user.focusCoins -= 50;
      user.streakFreezes = (user.streakFreezes ?? 0) + 1;
      await persistDb(db);
      res.json({ message: 'Streak Freeze Shield added to your inventory.', user: sanitizeUser(user) });
      return;
    }

    if (item === 'xp_boost') {
      if (user.focusCoins < 60) {
        res.status(400).json({ error: 'You need 60 Focus Coins for an XP Booster.' });
        return;
      }
      user.focusCoins -= 60;
      user.xp += 100;
      await persistDb(db);
      res.json({ message: '+100 XP applied to your Focus Rank!', user: sanitizeUser(user) });
      return;
    }

    if (item === 'break_pass') {
      if (user.settings.hardcoreMode) {
        res.status(403).json({ error: 'Break Passes are disabled while Hardcore Lock Mode is active.' });
        return;
      }
      const locked = user.activeLocks.find((l) => l.status === 'locked');
      if (!locked) {
        res.status(400).json({ error: 'You have no locked apps to apply a Break Pass to.' });
        return;
      }
      if (user.focusCoins < 40) {
        res.status(400).json({ error: 'You need 40 Focus Coins for a Break Pass.' });
        return;
      }
      user.focusCoins -= 40;
      locked.status = 'unlocked_by_task';
      await persistDb(db);
      res.json({ message: `Unlocked ${locked.appName} with an Earned Break Pass.`, user: sanitizeUser(user) });
      return;
    }

    res.status(400).json({ error: 'Unknown shop reward item.' });
  });

  // Live Leaderboard of actual registered users
  app.get('/api/leaderboard', requireAuth, async (req: AuthedRequest, res) => {
    const db = await loadDb();
    const currentUserId = req.authPayload!.sub;
    const sortBy = String(req.query.sortBy || 'xp');
    const tierFilter = String(req.query.tier || 'All');

    const tierCounts: Record<RankTierName, number> = {
      Bronze: 0,
      Silver: 0,
      Gold: 0,
      Platinum: 0,
      Diamond: 0,
      Master: 0,
      Grandmaster: 0,
      Emperor: 0,
    };

    const allUsers = Object.values(db.users).map((u) => {
      const info = computeLevelInfo(u.xp);
      tierCounts[info.rankTitle] += 1;
      return {
        userId: u.id,
        displayName: u.settings.displayName || u.name,
        xp: u.xp,
        level: info.level,
        rankTitle: info.rankTitle,
        streakDays: u.streakDays,
        totalTasksPassed: u.totalTasksPassed,
        totalMinutesLocked: u.totalMinutesLocked,
        bestScore: u.bestScore,
        isCurrentUser: u.id === currentUserId,
        createdAt: u.createdAt,
      };
    });

    allUsers.sort((a, b) => {
      if (sortBy === 'streak') {
        if (b.streakDays !== a.streakDays) return b.streakDays - a.streakDays;
      } else if (sortBy === 'tasks') {
        if (b.totalTasksPassed !== a.totalTasksPassed) return b.totalTasksPassed - a.totalTasksPassed;
      } else if (sortBy === 'minutes') {
        if (b.totalMinutesLocked !== a.totalMinutesLocked) return b.totalMinutesLocked - a.totalMinutesLocked;
      }
      if (b.xp !== a.xp) return b.xp - a.xp;
      if (b.totalTasksPassed !== a.totalTasksPassed) return b.totalTasksPassed - a.totalTasksPassed;
      return a.createdAt.localeCompare(b.createdAt);
    });

    const rankedAll: LeaderboardEntry[] = allUsers.map((entry, idx) => ({
      position: idx + 1,
      userId: entry.userId,
      displayName: entry.displayName,
      xp: entry.xp,
      level: entry.level,
      rankTitle: entry.rankTitle,
      streakDays: entry.streakDays,
      totalTasksPassed: entry.totalTasksPassed,
      totalMinutesLocked: entry.totalMinutesLocked,
      bestScore: entry.bestScore,
      isCurrentUser: entry.isCurrentUser,
    }));

    const currentUserEntry = rankedAll.find((e) => e.isCurrentUser);
    const validTiers = new Set(RANK_TIERS.map((t) => t.name));
    const filteredEntries =
      tierFilter !== 'All' && validTiers.has(tierFilter as RankTierName)
        ? rankedAll.filter((e) => e.rankTitle === tierFilter)
        : rankedAll;

    res.json({
      entries: filteredEntries,
      totalRegisteredUsers: rankedAll.length,
      tierCounts,
      currentUserPosition: currentUserEntry ? currentUserEntry.position : null,
    });
  });

  // Free Critique Studio (evaluate work even without an active lock)
  app.post('/api/critique/studio', requireAuth, async (req: AuthedRequest, res) => {
    try {
      const db = await loadDb();
      const user = db.users[req.authPayload!.email];
      if (!user) {
        res.status(404).json({ error: 'Account not found.' });
        return;
      }

      const { goalTitle, submissionText, imageBase64, imageMimeType } = req.body ?? {};
      const cleanTitle =
        typeof goalTitle === 'string' && goalTitle.trim().length >= 3
          ? goalTitle.trim().slice(0, 100)
          : 'General Work Review';
      const cleanText = typeof submissionText === 'string' ? submissionText.trim() : '';
      const hasImage = typeof imageBase64 === 'string' && imageBase64.length > 20;

      if (!cleanText && !hasImage) {
        res.status(400).json({ error: 'Provide work text or attach an image for Critique AI to review.' });
        return;
      }

      const allowedMimes = new Set(['image/png', 'image/jpeg', 'image/webp']);
      if (hasImage && !allowedMimes.has(imageMimeType)) {
        res.status(400).json({ error: 'Only PNG, JPEG, or WEBP images are supported.' });
        return;
      }

      const persona = PERSONA_DETAILS[user.settings.critiquePersona] ?? PERSONA_DETAILS.balanced;
      const activeGoals = normalizeSelectedGoals(
        user.settings.selectedGoals,
        user.settings.mainGoal
      );
      const activeGoalLabels = activeGoals
        .map((id) => SELF_IMPROVEMENT_GOALS.find((g) => g.id === id)?.label)
        .filter(Boolean)
        .join(', ');
      const ageContext = user.settings.age ? `User Age: ${user.settings.age}.` : '';
      const ai = createGeminiClient();

      const parts: Array<{ text: string } | { inlineData: { mimeType: string; data: string } }> = [
        {
          text: `Self-Improvement Focus: ${activeGoalLabels}\nTask / Goal Context: ${cleanTitle}\nPass Threshold: ${persona.threshold}/100\n\nWork / Action Log:\n${cleanText || '[See attached image]'}`,
        },
      ];
      if (hasImage) {
        parts.push({ inlineData: { mimeType: imageMimeType, data: imageBase64 } });
      }

      const aiResponse = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: { parts },
        config: {
          systemInstruction: `You are Critique AI, a sharp, practical self-improvement coach focused on Fitness, Looks & Style, Social Skills, and Mindset (${persona.label}: ${persona.summary}). The user's selected focus skills are: ${activeGoalLabels}. ${ageContext} Give concrete, realistic strengths, actionable critiques, and an objective score out of 100.`,
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              score: { type: Type.INTEGER },
              verdict: { type: Type.STRING },
              strengths: { type: Type.ARRAY, items: { type: Type.STRING } },
              critiques: { type: Type.ARRAY, items: { type: Type.STRING } },
              nextStep: { type: Type.STRING },
            },
            required: ['score', 'verdict', 'strengths', 'critiques', 'nextStep'],
          },
        },
      });

      const parsed = JSON.parse((aiResponse.text ?? '{}').trim()) as {
        score?: number;
        verdict?: string;
        strengths?: string[];
        critiques?: string[];
        nextStep?: string;
      };

      const score = Math.max(0, Math.min(100, Math.round(Number(parsed.score ?? 0))));
      const passed = score >= persona.threshold;

      const evaluation: CritiqueEvaluation = {
        passed,
        score,
        passThreshold: persona.threshold,
        verdict: parsed.verdict || 'Critique completed.',
        strengths: Array.isArray(parsed.strengths) ? parsed.strengths.slice(0, 4) : [],
        critiques: Array.isArray(parsed.critiques) ? parsed.critiques.slice(0, 4) : [],
        nextStep: parsed.nextStep || 'Apply the critique points above in your next draft.',
      };

      const xpEarned = passed ? 25 : 10;
      const coinsEarned = passed ? 10 : 0;
      user.xp += xpEarned;
      user.focusCoins += coinsEarned;
      user.bestScore = Math.max(user.bestScore, score);
      touchStreak(user);

      user.critiqueHistory.unshift({
        id: `crit-${crypto.randomUUID().slice(0, 8)}`,
        lockId: 'studio',
        appName: 'Critique Studio',
        taskTitle: cleanTitle,
        submissionSnippet: cleanText ? cleanText.slice(0, 140) : 'Image proof submitted',
        hasImageProof: hasImage,
        evaluation,
        xpEarned,
        coinsEarned,
        createdAt: new Date().toISOString(),
      });

      computeDailyQuests(user);
      await persistDb(db);

      res.json({ evaluation, xpEarned, coinsEarned, user: sanitizeUser(user) });
    } catch (err: unknown) {
      console.error('Critique studio error:', err);
      const message = err instanceof Error ? err.message : 'Unable to evaluate submission right now.';
      if (message.includes('GEMINI_API_KEY')) {
        res.status(503).json({
          error: 'Gemini API key is not configured yet. Add GEMINI_API_KEY in Settings > Secrets.',
        });
        return;
      }
      res.status(500).json({ error: `AI Critique failed: ${message}` });
    }
  });

  // Submit task work for Critique AI evaluation to unlock a locked app
  app.post('/api/critique/submit', requireAuth, async (req: AuthedRequest, res) => {
    try {
      const db = await loadDb();
      const user = db.users[req.authPayload!.email];
      if (!user) {
        res.status(404).json({ error: 'Account not found.' });
        return;
      }

      const { lockId, submissionText, imageBase64, imageMimeType } = req.body ?? {};
      const lock = user.activeLocks.find((l) => l.id === lockId && l.status === 'locked');
      if (!lock) {
        res.status(404).json({ error: 'This lock is no longer active.' });
        return;
      }

      const cleanText = typeof submissionText === 'string' ? submissionText.trim() : '';
      const hasImage = typeof imageBase64 === 'string' && imageBase64.length > 20;

      if (!cleanText && !hasImage) {
        res.status(400).json({ error: 'Provide written work or upload proof before requesting critique.' });
        return;
      }

      if (cleanText.length > 4000) {
        res.status(400).json({ error: 'Submission text must be under 4000 characters.' });
        return;
      }

      const allowedMimes = new Set(['image/png', 'image/jpeg', 'image/webp']);
      if (hasImage && !allowedMimes.has(imageMimeType)) {
        res.status(400).json({ error: 'Only PNG, JPEG, or WEBP proof images are supported.' });
        return;
      }

      const persona = PERSONA_DETAILS[user.settings.critiquePersona] ?? PERSONA_DETAILS.balanced;
      const activeGoals = normalizeSelectedGoals(
        user.settings.selectedGoals,
        user.settings.mainGoal
      );
      const activeGoalLabels = activeGoals
        .map((id) => SELF_IMPROVEMENT_GOALS.find((g) => g.id === id)?.label)
        .filter(Boolean)
        .join(', ');
      const ageContext = user.settings.age ? `User Age: ${user.settings.age}.` : '';
      const ai = createGeminiClient();

      const systemInstruction = `You are Critique AI, a sharp, fair personal development coach gating access to a locked distraction app (${lock.appName}).
Selected User Skills: ${activeGoalLabels}. ${ageContext}
Evaluation Persona: ${persona.label} (${persona.summary})
Pass Threshold: ${persona.threshold} out of 100.
Instructions:
- Evaluate whether the user genuinely completed their self-improvement task (Fitness, Looks & Style, Social Skills, or Mindset) with real effort, specificity, and alignment to the task prompt.
- Reject gibberish, single-sentence cop-outs, off-topic filler, or vague promises like "I did it".
- Give concrete, specific strengths and constructive, actionable critique points.
- Set "passed" to true ONLY if your objective score >= ${persona.threshold}.`;

      const promptText = `Locked App: ${lock.appName} (${lock.appCategory})
Task Title: ${lock.taskTitle}
Task Requirements: ${lock.taskPrompt}
Pass Threshold: ${persona.threshold}/100

User Submission:
${cleanText || '[See attached image proof]'}
`;

      const parts: Array<{ text: string } | { inlineData: { mimeType: string; data: string } }> = [
        { text: promptText },
      ];
      if (hasImage) {
        parts.push({
          inlineData: {
            mimeType: imageMimeType,
            data: imageBase64,
          },
        });
      }

      const aiResponse = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: { parts },
        config: {
          systemInstruction,
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              score: { type: Type.INTEGER },
              verdict: { type: Type.STRING },
              strengths: { type: Type.ARRAY, items: { type: Type.STRING } },
              critiques: { type: Type.ARRAY, items: { type: Type.STRING } },
              nextStep: { type: Type.STRING },
            },
            required: ['score', 'verdict', 'strengths', 'critiques', 'nextStep'],
          },
        },
      });

      const rawJson = (aiResponse.text ?? '{}').trim();
      const parsed = JSON.parse(rawJson) as {
        score?: number;
        verdict?: string;
        strengths?: string[];
        critiques?: string[];
        nextStep?: string;
      };

      const score = Math.max(0, Math.min(100, Math.round(Number(parsed.score ?? 0))));
      const passed = score >= persona.threshold;

      const evaluation: CritiqueEvaluation = {
        passed,
        score,
        passThreshold: persona.threshold,
        verdict: parsed.verdict || (passed ? 'Task verified and approved.' : 'Submission needs deeper effort.'),
        strengths: Array.isArray(parsed.strengths) ? parsed.strengths.slice(0, 4) : [],
        critiques: Array.isArray(parsed.critiques) ? parsed.critiques.slice(0, 4) : [],
        nextStep: parsed.nextStep || (passed ? 'Enjoy your earned access responsibly.' : 'Expand your response with concrete specifics.'),
      };

      lock.attemptsCount += 1;
      let xpEarned = 0;
      let coinsEarned = 0;

      if (passed) {
        lock.status = 'unlocked_by_task';
        const elapsedMinutes = Math.max(
          1,
          Math.round((Date.now() - new Date(lock.lockedAt).getTime()) / 60000)
        );
        user.totalMinutesLocked += Math.min(lock.durationMinutes, elapsedMinutes);
        user.totalTasksPassed += 1;
        user.bestScore = Math.max(user.bestScore, score);

        const hardcoreMultiplier = user.settings.hardcoreMode ? 1.25 : 1;
        xpEarned = Math.round((75 + score * 0.8 + lock.durationMinutes * 0.5) * hardcoreMultiplier);
        coinsEarned = Math.round((20 + score * 0.25) * hardcoreMultiplier);

        user.xp += xpEarned;
        user.focusCoins += coinsEarned;
        touchStreak(user);

        awardAchievement(user, 'first-unlock');
        if (score >= 90) awardAchievement(user, 'high-critique');
        if (user.settings.critiquePersona === 'strict') awardAchievement(user, 'strict-survivor');
      } else if (score >= 35) {
        xpEarned = 5;
        user.xp += xpEarned;
      }

      const historyEntry: CritiqueHistoryEntry = {
        id: `crit-${crypto.randomUUID().slice(0, 8)}`,
        lockId: lock.id,
        appName: lock.appName,
        taskTitle: lock.taskTitle,
        submissionSnippet: cleanText ? cleanText.slice(0, 140) : 'Image proof submitted',
        hasImageProof: hasImage,
        evaluation,
        xpEarned,
        coinsEarned,
        createdAt: new Date().toISOString(),
      };

      user.critiqueHistory.unshift(historyEntry);
      computeDailyQuests(user);
      await persistDb(db);

      res.json({
        evaluation,
        xpEarned,
        coinsEarned,
        user: sanitizeUser(user),
      });
    } catch (err: unknown) {
      console.error('Critique evaluation error:', err);
      const message = err instanceof Error ? err.message : 'Unable to evaluate submission right now.';
      if (message.includes('GEMINI_API_KEY')) {
        res.status(503).json({
          error: 'Gemini API key is not configured yet. Add GEMINI_API_KEY in Settings > Secrets.',
        });
        return;
      }
      res.status(500).json({
        error: `AI Critique failed: ${message}`,
      });
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        ws: { server: httpServer },
      },
      appType: 'spa',
    });
    app.use(express.static(path.resolve(process.cwd(), 'dev-dist'), { index: false }));
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log(`Enhance server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
