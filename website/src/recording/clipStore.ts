/**
 * Browser persistence for clips recorded with the gloves and assigned to dictionary classes.
 * Uses IndexedDB (clips are 100 KB – 2 MB each, too large for localStorage). Every call
 * fails soft: without storage (private mode, blocked site data) clips live for the session.
 */

import type { SignAnimationPayload } from '../types/lis';

const DB_NAME = 'lis-motion-studio';
const STORE = 'recorded-clips';

function openDb(): Promise<IDBDatabase | null> {
  return new Promise((resolve) => {
    try {
      if (typeof indexedDB === 'undefined') return resolve(null);
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = () => {
        if (!req.result.objectStoreNames.contains(STORE)) req.result.createObjectStore(STORE);
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
      req.onblocked = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

function run<T>(mode: IDBTransactionMode, fn: (store: IDBObjectStore) => IDBRequest<T>): Promise<T | null> {
  return openDb().then(
    (db) =>
      new Promise<T | null>((resolve) => {
        if (!db) return resolve(null);
        try {
          const tx = db.transaction(STORE, mode);
          const req = fn(tx.objectStore(STORE));
          req.onsuccess = () => resolve(req.result);
          req.onerror = () => resolve(null);
          tx.oncomplete = () => db.close();
        } catch {
          db.close();
          resolve(null);
        }
      }),
  );
}

export async function loadRecordedClips(): Promise<Record<string, SignAnimationPayload>> {
  const out: Record<string, SignAnimationPayload> = {};
  const db = await openDb();
  if (!db) return out;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE, 'readonly');
      const req = tx.objectStore(STORE).openCursor();
      req.onsuccess = () => {
        const cursor = req.result;
        if (cursor) {
          out[String(cursor.key)] = cursor.value as SignAnimationPayload;
          cursor.continue();
        } else {
          resolve(out);
        }
      };
      req.onerror = () => resolve(out);
      tx.oncomplete = () => db.close();
    } catch {
      db.close();
      resolve(out);
    }
  });
}

/** Returns true when the clip was stored durably. */
export async function saveRecordedClip(classId: string, payload: SignAnimationPayload): Promise<boolean> {
  const res = await run('readwrite', (s) => s.put(payload, classId));
  return res !== null;
}

export async function deleteRecordedClip(classId: string): Promise<void> {
  await run('readwrite', (s) => s.delete(classId));
}

/** Triggers a browser download of a JSON document. */
export function downloadJson(fileName: string, data: unknown): void {
  downloadText(fileName, JSON.stringify(data), 'application/json');
}

/** Triggers a browser download of a text file. */
export function downloadText(fileName: string, text: string, type = 'text/csv'): void {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
