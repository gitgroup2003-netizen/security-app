import React, { Suspense, lazy, useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Lock, Loader2, Eye, EyeOff } from 'lucide-react';
import { GitGroupLogo } from './GitGroupLogo';

// The main app (and all the heavy ML libraries it imports) is lazy-loaded so
// none of it is downloaded or executed until the correct password is entered.
const App = lazy(() => import('./App'));

// Only a salted SHA-256 hash of the password is stored here, never the
// password itself. NOTE: this is a front-end access gate. It keeps casual
// visitors out, but anyone determined can still read/modify client-side code.
// For real protection, verify the password on a server (see README).
const SALT = 'gitgroup-home-of-technology:';
const PASSWORD_HASH = '3a4b9b9aa6882712324725257ffb947b18aa32cea5c9f77a816f97f82aa08fae';
const SESSION_KEY = 'gitgroup_unlocked_v1';

const sha256Hex = async (text: string): Promise<string> => {
  const data = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest('SHA-256', data);
  return Array.from(new Uint8Array(digest)).map((b) => b.toString(16).padStart(2, '0')).join('');
};

type Stage = 'splash' | 'password' | 'unlocked';

export default function Gate() {
  const [stage, setStage] = useState<Stage>(() =>
    sessionStorage.getItem(SESSION_KEY) === '1' ? 'unlocked' : 'splash'
  );
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [failedAttempts, setFailedAttempts] = useState(0);
  const [lockedUntil, setLockedUntil] = useState(0);
  const [, forceTick] = useState(0);

  // Splash screen shows briefly, then moves to the password prompt.
  useEffect(() => {
    if (stage !== 'splash') return;
    const t = setTimeout(() => setStage('password'), 3200);
    return () => clearTimeout(t);
  }, [stage]);

  // Re-render while locked out so the countdown updates.
  useEffect(() => {
    if (lockedUntil <= Date.now()) return;
    const i = setInterval(() => forceTick((n) => n + 1), 500);
    return () => clearInterval(i);
  }, [lockedUntil]);

  const secondsLocked = Math.max(0, Math.ceil((lockedUntil - Date.now()) / 1000));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (checking || secondsLocked > 0) return;
    setChecking(true);
    setError(null);
    try {
      const hash = await sha256Hex(SALT + password);
      if (hash === PASSWORD_HASH) {
        sessionStorage.setItem(SESSION_KEY, '1');
        setStage('unlocked');
      } else {
        const attempts = failedAttempts + 1;
        setFailedAttempts(attempts);
        setPassword('');
        if (attempts >= 3) {
          // Escalating lockout after 3 wrong tries: 10s, 20s, 30s...
          setLockedUntil(Date.now() + (attempts - 2) * 10000);
          setError('Too many incorrect attempts.');
        } else {
          setError('Incorrect password.');
        }
      }
    } catch {
      setError('Secure check unavailable in this browser. Open the app over https or localhost.');
    } finally {
      setChecking(false);
    }
  };

  if (stage === 'unlocked') {
    return (
      <Suspense
        fallback={
          <div className="h-[100dvh] w-full bg-black text-white/70 flex items-center justify-center font-mono text-xs uppercase tracking-widest gap-3">
            <Loader2 className="w-4 h-4 animate-spin" /> Loading Git Group console...
          </div>
        }
      >
        <App />
      </Suspense>
    );
  }

  return (
    <div className="h-[100dvh] w-full bg-black text-white font-mono flex items-center justify-center overflow-hidden relative">
      <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.07)_0%,transparent_60%)]" />
      <div className="absolute inset-0 pointer-events-none bg-[linear-gradient(transparent_50%,rgba(0,0,0,0.3)_50%)] bg-[length:100%_4px]" />

      <AnimatePresence mode="wait">
        {stage === 'splash' && (
          <motion.div
            key="splash"
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.8 }}
            className="relative z-10 flex flex-col items-center text-center px-6"
          >
            <GitGroupLogo className="w-24 h-24 mb-6 drop-shadow-[0_0_18px_rgba(255,255,255,0.6)]" />
            <h1 className="text-4xl sm:text-5xl font-bold tracking-tighter drop-shadow-[0_0_10px_rgba(255,255,255,0.7)]">GIT GROUP</h1>
            <p className="mt-2 text-sm sm:text-base uppercase tracking-[0.35em] text-white/80">Home of Technology</p>
            <div className="mt-8 h-px w-40 bg-white/30" />
            <p className="mt-4 text-[11px] uppercase tracking-widest text-white/50">CEO</p>
            <p className="text-sm sm:text-base tracking-wider text-white/90">Frank Ssemakula</p>
          </motion.div>
        )}

        {stage === 'password' && (
          <motion.form
            key="password"
            onSubmit={handleSubmit}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.6 }}
            className="relative z-10 w-full max-w-sm px-6 flex flex-col items-center"
          >
            <GitGroupLogo className="w-14 h-14 mb-4 opacity-90" />
            <h2 className="text-lg font-bold tracking-widest uppercase">Git Group</h2>
            <p className="text-[10px] uppercase tracking-[0.3em] text-white/50 mb-8">Home of Technology</p>

            <div className="w-full border border-white/20 bg-black/50 backdrop-blur-md p-5 space-y-4">
              <label className="flex items-center gap-2 text-[10px] uppercase tracking-widest text-white/60">
                <Lock className="w-3 h-3" /> Authorized access only
              </label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  autoFocus
                  autoComplete="off"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter password"
                  disabled={secondsLocked > 0}
                  className="w-full bg-white/5 border border-white/20 px-3 py-2.5 pr-10 text-sm text-white placeholder:text-white/30 focus:outline-none focus:border-white/60 disabled:opacity-50"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((v) => !v)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-white/40 hover:text-white/80"
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>

              {(error || secondsLocked > 0) && (
                <p className="text-[11px] text-red-400">
                  {error}
                  {secondsLocked > 0 && ` Try again in ${secondsLocked}s.`}
                </p>
              )}

              <button
                type="submit"
                disabled={checking || secondsLocked > 0 || password.length === 0}
                className="w-full flex justify-center items-center gap-2 px-4 py-3 text-xs font-bold uppercase tracking-widest border border-white bg-white/10 hover:bg-white/20 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                {checking ? <><Loader2 className="w-4 h-4 animate-spin" /> Verifying...</> : 'Unlock'}
              </button>
            </div>

            <p className="mt-6 text-[10px] text-white/30 uppercase tracking-widest">CEO · Frank Ssemakula</p>
          </motion.form>
        )}
      </AnimatePresence>
    </div>
  );
}
