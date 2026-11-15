import React, { useRef, useState } from 'react';
import { Upload, ArrowRight, UserRound } from 'lucide-react';
import { SIGN_LANGUAGES, SOURCE_LANGUAGES, type LanguageSettings } from '../i18n/languages';
import { BUILTIN_AVATARS, DEFAULT_AVATAR_ID, type AvatarChoice } from '../avatars/avatarCatalog';

const FILE_OPTION = '__file__';

export type AppViewMode = 'interpreter' | 'recordings' | 'qa' | 'dictionary' | 'architecture';

interface TopBarProps {
  activeView: AppViewMode;
  onChangeView: (view: AppViewMode) => void;
  languages: LanguageSettings;
  onChangeLanguages: (next: LanguageSettings) => void;
  avatar: AvatarChoice;
  onChangeAvatar: (next: AvatarChoice) => void;
}

const NAV_ITEMS: Array<{ id: AppViewMode; label: string }> = [
  { id: 'interpreter', label: 'Interpreter' },
  { id: 'recordings', label: 'Recordings' },
  { id: 'qa', label: 'QA comparison' },
  { id: 'dictionary', label: 'Dictionary' },
  { id: 'architecture', label: 'Architecture' },
];

const selectClass =
  'rounded-md border border-slate-300 bg-white px-2 py-1 text-xs font-medium text-slate-900 focus:border-blue-600 focus:outline-none cursor-pointer';

export const TopBar: React.FC<TopBarProps> = ({
  activeView,
  onChangeView,
  languages,
  onChangeLanguages,
  avatar,
  onChangeAvatar,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  // Last local file, kept in the list so the user can switch back to it.
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedFile(file);
      onChangeAvatar({ kind: 'file', file });
    }
    e.target.value = '';
  };

  const handleSelectAvatar = (value: string) => {
    if (value === FILE_OPTION) {
      if (uploadedFile) onChangeAvatar({ kind: 'file', file: uploadedFile });
      return;
    }
    onChangeAvatar({ kind: 'builtin', id: value });
  };

  return (
    <header className="h-14 shrink-0 bg-white border-b border-slate-200 px-6 flex items-center justify-between gap-6">
      {/* Zone 1: brand wordmark */}
      <a
        href="#interpreter"
        onClick={(e) => {
          e.preventDefault();
          onChangeView('interpreter');
        }}
        className="font-display text-lg font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0"
      >
        LIS Motion Studio
      </a>

      {/* Zone 2: navigation */}
      <nav aria-label="Main navigation" className="flex items-center gap-6 overflow-x-auto">
        {NAV_ITEMS.map((item) => {
          const isActive = activeView === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onChangeView(item.id)}
              aria-current={isActive ? 'page' : undefined}
              className={`py-1 text-sm font-medium whitespace-nowrap shrink-0 border-b-2 transition-colors cursor-pointer ${
                isActive
                  ? 'border-blue-600 text-slate-900'
                  : 'border-transparent text-slate-500 hover:text-slate-900'
              }`}
            >
              {item.label}
            </button>
          );
        })}
      </nav>

      {/* Zone 3: languages + avatar */}
      <div className="flex items-center gap-3 shrink-0">
        <div className="flex items-center gap-1.5" role="group" aria-label="Translation languages">
          <label htmlFor="source-language" className="sr-only">
            Source language
          </label>
          <select
            id="source-language"
            value={languages.sourceLanguage}
            onChange={(e) => onChangeLanguages({ ...languages, sourceLanguage: e.target.value })}
            className={selectClass}
            title="Language of the phrase you type"
          >
            {SOURCE_LANGUAGES.map((l) => (
              <option key={l.code} value={l.code}>
                {l.label}
              </option>
            ))}
          </select>
          <ArrowRight className="w-3.5 h-3.5 text-slate-400" aria-hidden="true" />
          <label htmlFor="sign-language" className="sr-only">
            Sign language
          </label>
          <select
            id="sign-language"
            value={languages.signLanguage}
            onChange={(e) => onChangeLanguages({ ...languages, signLanguage: e.target.value })}
            className={selectClass}
            title="Target sign language"
          >
            {SIGN_LANGUAGES.map((l) => (
              <option key={l.code} value={l.code}>
                {l.code}
              </option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-1.5" role="group" aria-label="Avatar">
          <UserRound className="w-3.5 h-3.5 text-slate-400" aria-hidden="true" />
          <label htmlFor="avatar-select" className="sr-only">
            Avatar
          </label>
          <select
            id="avatar-select"
            value={avatar.kind === 'file' ? FILE_OPTION : avatar.id}
            onChange={(e) => handleSelectAvatar(e.target.value)}
            className={`${selectClass} max-w-[11rem]`}
            title="Avatar shown in the viewport"
          >
            {BUILTIN_AVATARS.map((a) => (
              <option key={a.id} value={a.id}>
                {a.id === DEFAULT_AVATAR_ID ? `${a.label} (default)` : a.label}
              </option>
            ))}
            {uploadedFile && <option value={FILE_OPTION}>{uploadedFile.name}</option>}
          </select>
          <input
            ref={fileInputRef}
            type="file"
            accept=".vrm,.glb,.gltf"
            onChange={handleFileChange}
            className="sr-only"
            aria-label="Select a VRM or GLB avatar file"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            title="Load a local VRM / GLB avatar"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-slate-900 hover:bg-slate-800 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Upload className="w-3.5 h-3.5" aria-hidden="true" />
            Load VRM / GLB
          </button>
        </div>
      </div>
    </header>
  );
};
