import test from 'node:test';
import assert from 'node:assert/strict';
import {
  AvatarNotFoundError,
  BUILTIN_AVATARS,
  DEFAULT_AVATAR_ID,
  fetchAvatarBuffer,
  findBuiltinAvatar,
  looksLikeGltf,
  PROCEDURAL_AVATAR_ID,
} from './avatarCatalog';

const bytes = (s: string) => new TextEncoder().encode(s).buffer as ArrayBuffer;

test('the default avatar is woman.glb and the procedural avatar is always available', () => {
  assert.equal(DEFAULT_AVATAR_ID, 'woman');
  assert.match(findBuiltinAvatar(DEFAULT_AVATAR_ID).url ?? '', /\/assets\/woman\.glb$/);
  assert.equal(findBuiltinAvatar(PROCEDURAL_AVATAR_ID).url, undefined);
  assert.equal(findBuiltinAvatar('unknown').id, DEFAULT_AVATAR_ID);
  assert.equal(new Set(BUILTIN_AVATARS.map((a) => a.id)).size, BUILTIN_AVATARS.length);
});

test('looksLikeGltf accepts GLB and .gltf JSON, rejects the HTML fallback page', () => {
  assert.ok(looksLikeGltf(bytes('glTF\u0002\u0000\u0000\u0000')));
  assert.ok(looksLikeGltf(bytes('  {"asset": {"version": "2.0"}, "scenes": []}')));
  assert.ok(!looksLikeGltf(bytes('<!doctype html><html></html>')));
  assert.ok(!looksLikeGltf(bytes('{"error": 1}')));
  assert.ok(!looksLikeGltf(new ArrayBuffer(0)));
});

test('fetchAvatarBuffer reports a missing file as AvatarNotFoundError', async () => {
  const realFetch = globalThis.fetch;
  try {
    globalThis.fetch = (async () => new Response('<!doctype html>', { status: 200 })) as typeof fetch;
    await assert.rejects(fetchAvatarBuffer('/assets/woman.glb'), AvatarNotFoundError);
    globalThis.fetch = (async () => new Response('', { status: 404 })) as typeof fetch;
    await assert.rejects(fetchAvatarBuffer('/assets/woman.glb'), AvatarNotFoundError);
    globalThis.fetch = (async () => new Response(bytes('glTF\u0002\u0000\u0000\u0000'))) as typeof fetch;
    assert.equal((await fetchAvatarBuffer('/assets/woman.glb')).byteLength, 8);
  } finally {
    globalThis.fetch = realFetch;
  }
});
