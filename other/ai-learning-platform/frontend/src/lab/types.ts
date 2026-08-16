// Math Lab types — mirror backend lab/modules.py + compute outputs.

export type ControlType = "select" | "range" | "toggle" | "action";

export interface LabControl {
  key: string;
  label: string;
  type: ControlType;
  options?: string[];
  min?: number;
  max?: number;
  step?: number;
  default?: number | string | boolean;
}

export interface LabModule {
  id: string;
  title: string;
  subtitle: string;
  icon: string;
  category: string;
  blurb: string;
  group?: string;
  question?: string;
  next_question?: string;
  next_experiment?: string;
  controls: LabControl[];
}

export type LabParams = Record<string, unknown>;

// Generic compute result — each module adds its own fields.
export type LabResult = Record<string, unknown>;
