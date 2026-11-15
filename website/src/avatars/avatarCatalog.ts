/**
 * Avatar catalog and avatar choice.
 *
 * Built-in avatars are GLB/VRM files served from `public/assets/` (URL `/assets/<file>`), plus
 * the procedural avatar generated in code. The user can also load a local VRM/GLB file.
 * The built-in choice is remembered per browser; a local file is kept for the session only.
 *
 * To add a built-in avatar: put the file in `public/assets/` and add an entry to BUILTIN_AVATARS.
 */

import { useEffect, useState } from 'react';

export interface BuiltinAvatar {
  id: string;
  label: string;
  /** GLB/VRM URL. Undefined for the procedural avatar. */
  url?: string;
}

export const PROCEDURAL_AVATAR_ID = 'procedural';

export const BUILTIN_AVATARS: BuiltinAvatar[] = [
  { id: 'woman', label: 'Woman', url: '/assets/woman.glb' },
  { id: PROCEDURAL_AVATAR_ID, label: 'Procedural' },
];

export const DEFAULT_AVATAR_ID = 'woman';

export type AvatarChoice = { kind: 'builtin'; id: string } | { kind: 'file'; file: File };

export function findBuiltinAvatar(id: string): BuiltinAvatar {
  return (
    BUILTIN_AVATARS.find((a) => a.id === id) ??
    BUILTIN_AVATARS.find((a) => a.id === DEFAULT_AVATAR_ID) ??
    BUILTIN_AVATARS[0]
  );
}

/** Display name of a choice. */
export function avatarChoiceName(choice: AvatarChoice): string {
  return choice.kind === 'file' ? choice.file.name : findBuiltinAvatar(choice.id).label;
}

/** The avatar file is not at its URL (404, or the dev server answered with the HTML page). */
export class AvatarNotFoundError extends Error {
  constructor(public readonly url: string) {
    super(`Avatar file not found: ${url}`);
    this.name = 'AvatarNotFoundError';
  }
}

/**
 * Downloads a built-in avatar file. Only real glTF content is accepted: binary GLB/VRM
 * (magic "glTF") or JSON .gltf. Anything else counts as a missing file.
 */
export async function fetchAvatarBuffer(url: string, signal?: AbortSignal): Promise<ArrayBuffer> {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new AvatarNotFoundError(url);
  const buffer = await res.arrayBuffer();
  if (!looksLikeGltf(buffer)) throw new AvatarNotFoundError(url);
  return buffer;
}

export function looksLikeGltf(buffer: ArrayBuffer): boolean {
  if (buffer.byteLength < 4) return false;
  const head = new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 4096));
  if (head[0] === 0x67 && head[1] === 0x6c && head[2] === 0x54 && head[3] === 0x46) return true; // "glTF"
  // JSON .gltf: a JSON object with an "asset" block.
  const text = new TextDecoder().decode(head).trimStart();
  return text.startsWith('{') && text.includes('"asset"');
}

const STORAGE_KEY = 'lis-motion-studio.avatar';

function readStoredId(): string {
  try {
    const id = localStorage.getItem(STORAGE_KEY);
    if (id && BUILTIN_AVATARS.some((a) => a.id === id)) return id;
  } catch {
    // storage unavailable
  }
  return DEFAULT_AVATAR_ID;
}

/** Avatar choice; the built-in id is remembered per browser. */
export function useAvatarChoice(): [AvatarChoice, (next: AvatarChoice) => void] {
  const [choice, setChoice] = useState<AvatarChoice>(() => ({ kind: 'builtin', id: readStoredId() }));
  useEffect(() => {
    if (choice.kind !== 'builtin') return;
    try {
      localStorage.setItem(STORAGE_KEY, choice.id);
    } catch {
      // storage unavailable: keep the choice for this session only
    }
  }, [choice]);
  return [choice, setChoice];
}
