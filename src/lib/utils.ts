import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(num: number): string {
  return new Intl.NumberFormat('en-IN').format(num);
}

export function formatCompactNumber(num: number): string {
  if (num >= 10000000) {
    return (num / 10000000).toFixed(1) + ' Cr';
  }
  if (num >= 100000) {
    return (num / 100000).toFixed(1) + ' L';
  }
  if (num >= 1000) {
    return (num / 1000).toFixed(1) + ' K';
  }
  return num.toString();
}

export function debounce<T extends (...args: unknown[]) => unknown>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout>;
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
}

export function generateId(prefix = 'id'): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

export function getInitials(name: string): string {
  return name
    .split(' ')
    .map(part => part[0])
    .join('')
    .toUpperCase()
    .slice(0, 2);
}

export function truncate(str: string, length: number): string {
  if (str.length <= length) return str;
  return str.slice(0, length) + '...';
}

export function isValidUrl(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

export function getFileSizeString(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function isImageFile(file: File): boolean {
  return file.type.startsWith('image/');
}

export function isVideoFile(file: File): boolean {
  return file.type.startsWith('video/');
}

export function isValidImageFile(file: File): boolean {
  const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
  return validTypes.includes(file.type) && file.size <= 10 * 1024 * 1024;
}

export function isValidVideoFile(file: File): boolean {
  const validTypes = ['video/mp4', 'video/quicktime'];
  return validTypes.includes(file.type) && file.size <= 50 * 1024 * 1024;
}

export function createObjectURL(file: File): string {
  return URL.createObjectURL(file);
}

export function revokeObjectURL(url: string): void {
  URL.revokeObjectURL(url);
}

export function downloadAsFile(content: string, filename: string, type = 'text/plain'): void {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export function parseJSONSafely<T>(json: string, fallback: T): T {
  try {
    return JSON.parse(json);
  } catch {
    return fallback;
  }
}

export function deepClone<T>(obj: T): T {
  return JSON.parse(JSON.stringify(obj));
}

export function groupBy<T>(array: T[], key: keyof T): Record<string, T[]> {
  return array.reduce((acc, item) => {
    const groupKey = String(item[key]);
    if (!acc[groupKey]) {
      acc[groupKey] = [];
    }
    acc[groupKey].push(item);
    return acc;
  }, {} as Record<string, T[]>);
}

export function sortBy<T>(array: T[], key: keyof T, direction: 'asc' | 'desc' = 'asc'): T[] {
  return [...array].sort((a, b) => {
    const aVal = a[key];
    const bVal = b[key];
    if (aVal < bVal) return direction === 'asc' ? -1 : 1;
    if (aVal > bVal) return direction === 'asc' ? 1 : -1;
    return 0;
  });
}

export function filterBy<T>(array: T[], predicates: Partial<Record<keyof T, unknown>>): T[] {
  return array.filter(item =>
    Object.entries(predicates).every(([key, value]) => {
      if (value === undefined || value === '') return true;
      const itemValue = item[key as keyof T];
      if (Array.isArray(value)) {
        return value.includes(itemValue);
      }
      return itemValue === value;
    })
  );
}

// ─── API Client ─────────────────────────────────────────────────────────────
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface DamageDetection {
  damage_type: string;
  confidence: number;
  bounding_box_xyxy: number[];
}

export interface ViewResult {
  detection_count: number;
  detections: DamageDetection[];
  annotated_image_base64?: string;
}

export interface AssessmentResponse {
  assessment_id: string;
  created_at: string;
  workflow: string;
  model_name: string;
  confidence_threshold: number;
  image_size: number;
  views: Record<string, ViewResult>;
  detection_counts_by_class: Record<string, number>;
  total_detections: number;
  review_required: boolean;
  review_flags: Array<{ code: string; view: string; message: string }>;
  validation: Record<string, any>;
  limitations: string[];
  estimated_repair_cost_min: number;
  estimated_repair_cost_max: number;
  settlement_offer: number;
  settlement_min: number;
  settlement_max: number;
  severity: 'minor' | 'moderate' | 'severe' | 'uncertain';
  primary_component: string;
  damage_summary: string;
}

export interface GeneralAssessmentResponse {
  assessment_id: string;
  created_at: string;
  model_name: string;
  confidence_threshold: number;
  image_size: number;
  images_analyzed: number;
  results: Array<{
    file_index: number;
    filename: string;
    detection_count: number;
    detections: DamageDetection[];
    annotated_image_base64?: string;
    error?: string;
  }>;
  total_detections: number;
  estimated_repair_cost_min: number;
  estimated_repair_cost_max: number;
  settlement_offer: number;
  settlement_min: number;
  settlement_max: number;
  severity: 'minor' | 'moderate' | 'severe' | 'uncertain';
  primary_component: string;
  damage_summary: string;
  review_required: boolean;
}

export interface ModelInfo {
  model_name: string;
  model_type: string;
  task: string;
  classes: string[];
}

export interface HealthCheck {
  status: string;
  service: string;
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`);
  }
  return response.json();
}

export async function checkHealth(): Promise<HealthCheck> {
  const response = await fetch(`${API_BASE_URL}/health`);
  return handleResponse(response);
}

export async function getModelInfo(): Promise<ModelInfo> {
  const response = await fetch(`${API_BASE_URL}/model/info`);
  return handleResponse(response);
}

export interface FourViewAssessmentRequest {
  top: File;
  bottom: File;
  left: File;
  right: File;
  confidence_threshold?: number;
  image_size?: number;
}

export async function assessFourView(request: FourViewAssessmentRequest): Promise<AssessmentResponse> {
  const formData = new FormData();
  formData.append('top', request.top);
  formData.append('bottom', request.bottom);
  formData.append('left', request.left);
  formData.append('right', request.right);
  formData.append('confidence_threshold', String(request.confidence_threshold ?? 0.25));
  formData.append('image_size', String(request.image_size ?? 640));

  const response = await fetch(`${API_BASE_URL}/assess/four-view`, {
    method: 'POST',
    body: formData,
  });
  return handleResponse(response);
}

export interface GeneralAssessmentRequest {
  files: File[];
  confidence_threshold?: number;
  image_size?: number;
}

export async function assessGeneral(request: GeneralAssessmentRequest): Promise<GeneralAssessmentResponse> {
  const formData = new FormData();
  request.files.forEach(file => formData.append('files', file));
  formData.append('confidence_threshold', String(request.confidence_threshold ?? 0.25));
  formData.append('image_size', String(request.image_size ?? 640));

  const response = await fetch(`${API_BASE_URL}/assess/general`, {
    method: 'POST',
    body: formData,
  });
  return handleResponse(response);
}

// Convert base64 to data URL for display
export function base64ToDataUrl(base64: string, mimeType = 'image/jpeg'): string {
  return `data:${mimeType};base64,${base64}`;
}

// ─── Hybrid AI Assessment ───────────────────────────────────────────────────

export interface HybridAssessmentRequest {
  vehicle: {
    make: string;
    model: string;
    year?: number;
    variant?: string;
  };
  views: Record<string, string>; // base64 encoded images
  detections: Record<string, DamageDetection[]>;
  confidence_threshold?: number;
  image_size?: number;
}

export interface HybridDamageItem {
  component: string;
  view: string;
  detectionIds: string[];
  damageType: string;
  severity: 'minor' | 'moderate' | 'severe' | 'uncertain';
  evidence: string[];
  confidence: number;
  recommendation: 'repair' | 'replace' | 'inspect' | 'undetermined';
  reasoning: string;
  assumptions: string[];
}

export interface HybridAIObservation {
  component: string;
  view: string;
  description: string;
  severity: 'minor' | 'moderate' | 'severe' | 'uncertain';
  confidence: number;
  reasoning: string;
  boundingBoxHint?: { x: number; y: number; width: number; height: number };
}

export interface HybridDuplicateWarning {
  detectionIds: string[];
  views: string[];
  message: string;
}

export interface HybridAssessmentResponse {
  assessment_id: string;
  created_at: string;
  workflow: string;
  model_name: string;
  ai_provider: string;
  ai_model: string;
  damage_items: HybridDamageItem[];
  ai_observations: HybridAIObservation[];
  duplicate_warnings: HybridDuplicateWarning[];
  overall_reasoning: string;
  confidence: number;
  limitations: string[];
  estimated_repair_cost_min: number;
  estimated_repair_cost_max: number;
  settlement_offer: number;
  settlement_min: number;
  settlement_max: number;
  severity: 'minor' | 'moderate' | 'severe' | 'uncertain';
  primary_component: string;
  damage_summary: string;
  review_required: boolean;
}

export async function assessHybrid(request: HybridAssessmentRequest): Promise<HybridAssessmentResponse> {
  const response = await fetch(`${API_BASE_URL}/assess/hybrid`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  return handleResponse(response);
}

export interface AIProviderStatus {
  provider: string;
  gemini_configured: boolean;
  gemini_model: string;
  openai_configured: boolean;
  openai_model: string;
  hybrid_available: boolean;
}

export async function getAIProviderStatus(): Promise<AIProviderStatus> {
  const response = await fetch(`${API_BASE_URL}/ai/provider/status`);
  return handleResponse(response);
}