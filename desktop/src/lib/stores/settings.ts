// Application settings with localStorage persistence
import { writable, derived } from 'svelte/store';
import { browser } from '$app/environment';

export interface AppSettings {
  theme: 'dark' | 'light' | 'system';
  language: 'en' | 'pt' | 'es';
  juliaServerUrl: string;
  ollamaServerUrl: string;
  autoConnect: boolean;
  showWelcome: boolean;
  recentWorkspaces: string[];
  visualization: {
    showGrid: boolean;
    showAxes: boolean;
    antialiasing: boolean;
    defaultHeatmap: string;
  };
  export: {
    defaultFormat: 'stl' | 'gcode' | 'obj';
    defaultQuality: 'low' | 'medium' | 'high';
    outputDirectory: string;
  };
  ai: {
    defaultAgent: 'design' | 'analysis' | 'synthesis';
    ollamaModel: string;
    streamResponses: boolean;
  };
}

const defaultSettings: AppSettings = {
  theme: 'dark',
  language: 'en',
  juliaServerUrl: 'http://localhost:8081',
  ollamaServerUrl: 'http://localhost:11434',
  autoConnect: true,
  showWelcome: true,
  recentWorkspaces: [],
  visualization: {
    showGrid: true,
    showAxes: true,
    antialiasing: true,
    defaultHeatmap: 'porosity',
  },
  export: {
    defaultFormat: 'stl',
    defaultQuality: 'medium',
    outputDirectory: '',
  },
  ai: {
    defaultAgent: 'design',
    ollamaModel: 'qwen2.5:7b',
    streamResponses: true,
  },
};

function loadSettings(): AppSettings {
  if (!browser) return defaultSettings;

  try {
    const stored = localStorage.getItem('darwin-scaffold-settings');
    if (stored) {
      const parsed = JSON.parse(stored);
      // Merge with defaults to handle new settings
      return { ...defaultSettings, ...parsed };
    }
  } catch (e) {
    console.error('Failed to load settings:', e);
  }

  return defaultSettings;
}

function saveSettings(settings: AppSettings) {
  if (!browser) return;

  try {
    localStorage.setItem('darwin-scaffold-settings', JSON.stringify(settings));
  } catch (e) {
    console.error('Failed to save settings:', e);
  }
}

function createSettingsStore() {
  const { subscribe, set, update } = writable<AppSettings>(loadSettings());

  return {
    subscribe,
    set: (value: AppSettings) => {
      saveSettings(value);
      set(value);
    },
    update: (fn: (settings: AppSettings) => AppSettings) => {
      update((settings) => {
        const newSettings = fn(settings);
        saveSettings(newSettings);
        return newSettings;
      });
    },
    reset: () => {
      saveSettings(defaultSettings);
      set(defaultSettings);
    },
  };
}

export const settings = createSettingsStore();

// Derived stores for easy access
export const theme = derived(settings, ($settings) => $settings.theme);
export const language = derived(settings, ($settings) => $settings.language);

// Theme helper - resolves 'system' to actual theme
export const resolvedTheme = derived(settings, ($settings) => {
  if ($settings.theme === 'system') {
    if (browser) {
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    return 'dark';
  }
  return $settings.theme;
});

// Actions
export function setTheme(theme: 'dark' | 'light' | 'system') {
  settings.update((s) => ({ ...s, theme }));
}

export function setLanguage(language: 'en' | 'pt' | 'es') {
  settings.update((s) => ({ ...s, language }));
}

export function addRecentWorkspace(workspaceId: string) {
  settings.update((s) => {
    const recent = [workspaceId, ...s.recentWorkspaces.filter((id) => id !== workspaceId)].slice(0, 10);
    return { ...s, recentWorkspaces: recent };
  });
}

export function updateVisualizationSettings(viz: Partial<AppSettings['visualization']>) {
  settings.update((s) => ({
    ...s,
    visualization: { ...s.visualization, ...viz },
  }));
}

export function updateExportSettings(exp: Partial<AppSettings['export']>) {
  settings.update((s) => ({
    ...s,
    export: { ...s.export, ...exp },
  }));
}

export function updateAISettings(ai: Partial<AppSettings['ai']>) {
  settings.update((s) => ({
    ...s,
    ai: { ...s.ai, ...ai },
  }));
}
