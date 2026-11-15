/**
 * Base URL of the site ("./" in the build, see vite.config.ts), so data files load from the
 * same folder as the page on any host path (GitHub Pages serves it under /<repo>/).
 * "/" outside Vite (node tests).
 */
export const BASE_URL: string = (import.meta as { env?: { BASE_URL?: string } }).env?.BASE_URL ?? '/';
