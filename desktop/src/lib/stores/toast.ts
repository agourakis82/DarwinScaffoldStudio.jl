// Toast notification system
import { writable, derived } from 'svelte/store';

export interface Toast {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
  duration: number;
  dismissible: boolean;
}

const toastStore = writable<Toast[]>([]);

export const toasts = {
  subscribe: toastStore.subscribe,

  add: (toast: Omit<Toast, 'id'>) => {
    const id = crypto.randomUUID();
    const newToast: Toast = { ...toast, id };

    toastStore.update((all) => [newToast, ...all]);

    if (toast.duration > 0) {
      setTimeout(() => {
        toasts.dismiss(id);
      }, toast.duration);
    }

    return id;
  },

  dismiss: (id: string) => {
    toastStore.update((all) => all.filter((t) => t.id !== id));
  },

  clear: () => {
    toastStore.set([]);
  },
};

// Convenience functions
export function showSuccess(title: string, message?: string) {
  return toasts.add({
    type: 'success',
    title,
    message,
    duration: 4000,
    dismissible: true,
  });
}

export function showError(title: string, message?: string) {
  return toasts.add({
    type: 'error',
    title,
    message,
    duration: 6000,
    dismissible: true,
  });
}

export function showWarning(title: string, message?: string) {
  return toasts.add({
    type: 'warning',
    title,
    message,
    duration: 5000,
    dismissible: true,
  });
}

export function showInfo(title: string, message?: string) {
  return toasts.add({
    type: 'info',
    title,
    message,
    duration: 4000,
    dismissible: true,
  });
}

// For operations that might take a while
export function showLoading(title: string, message?: string) {
  return toasts.add({
    type: 'info',
    title,
    message,
    duration: 0, // Don't auto-dismiss
    dismissible: false,
  });
}
