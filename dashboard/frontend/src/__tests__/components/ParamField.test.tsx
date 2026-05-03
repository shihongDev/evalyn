/**
 * Tests for ParamField - one block per `kind`.
 *
 * Verifies:
 *   - Each kind renders the expected control.
 *   - onChange fires the correct payload for the kind.
 *   - Required + advanced markers render.
 */

import { describe, expect, test, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import ParamField from '../../views/ParamField';
import type { ParamSchema } from '../../types/catalog';

const renderField = (param: ParamSchema, value: unknown = '') => {
  const onChange = vi.fn();
  const utils = render(<ParamField param={param} value={value} onChange={onChange} />);
  return { ...utils, onChange };
};

describe('ParamField - bool', () => {
  test('renders default / on / off buttons', () => {
    renderField({ name: 'verbose', kind: 'bool', default: false });
    expect(screen.getByRole('button', { name: 'default' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'on' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'off' })).toBeInTheDocument();
  });

  test('clicking on calls onChange(true)', () => {
    const { onChange } = renderField(
      { name: 'verbose', kind: 'bool', default: false },
      false,
    );
    fireEvent.click(screen.getByRole('button', { name: 'on' }));
    expect(onChange).toHaveBeenCalledWith(true);
  });

  test('clicking off calls onChange(false)', () => {
    const { onChange } = renderField(
      { name: 'verbose', kind: 'bool', default: false },
      true,
    );
    fireEvent.click(screen.getByRole('button', { name: 'off' }));
    expect(onChange).toHaveBeenCalledWith(false);
  });

  test('clicking default calls onChange with the param default (so buildCli skips the flag)', () => {
    const { onChange } = renderField(
      { name: 'verbose', kind: 'bool', default: false },
      true,
    );
    fireEvent.click(screen.getByRole('button', { name: 'default' }));
    // default = false on this param, so "default" mode stores false to make
    // buildCli's `cur === def` skip path fire.
    expect(onChange).toHaveBeenCalledWith(false);
  });

  test('default mode reflected on first render when value equals param.default', () => {
    renderField({ name: 'verbose', kind: 'bool', default: false }, false);
    const def = screen.getByRole('button', { name: 'default' });
    expect(def).toHaveAttribute('aria-pressed', 'true');
  });

  test('on mode reflected when value differs from default', () => {
    renderField({ name: 'verbose', kind: 'bool', default: false }, true);
    const on = screen.getByRole('button', { name: 'on' });
    expect(on).toHaveAttribute('aria-pressed', 'true');
  });

  test('shows resolved default below the segment control', () => {
    renderField({ name: 'verbose', kind: 'bool', default: true });
    expect(screen.getByText(/default = true/)).toBeInTheDocument();
  });
});

describe('ParamField - string', () => {
  test('renders a text input', () => {
    renderField({ name: 'name', kind: 'string' }, 'hello');
    expect(screen.getByDisplayValue('hello')).toBeInTheDocument();
  });

  test('typing fires onChange with new value', () => {
    const { onChange } = renderField({ name: 'name', kind: 'string' }, '');
    const input = screen.getByLabelText(/name/i) as HTMLInputElement;
    fireEvent.change(input, { target: { value: 'world' } });
    expect(onChange).toHaveBeenCalledWith('world');
  });
});

describe('ParamField - number', () => {
  test('renders a number input', () => {
    renderField({ name: 'workers', kind: 'number', default: 4 }, 4);
    const input = screen.getByLabelText(/workers/i) as HTMLInputElement;
    expect(input.type).toBe('number');
  });

  test('numeric input parses to Number', () => {
    const { onChange } = renderField(
      { name: 'workers', kind: 'number', default: 4 },
      4,
    );
    const input = screen.getByLabelText(/workers/i) as HTMLInputElement;
    fireEvent.change(input, { target: { value: '8' } });
    expect(onChange).toHaveBeenCalledWith(8);
  });

  test('empty string passes through as ""', () => {
    const { onChange } = renderField(
      { name: 'workers', kind: 'number', default: 4 },
      4,
    );
    const input = screen.getByLabelText(/workers/i) as HTMLInputElement;
    fireEvent.change(input, { target: { value: '' } });
    expect(onChange).toHaveBeenCalledWith('');
  });

  test('number kind WITH range renders slider + unit suffix', () => {
    const { container } = renderField(
      {
        name: 'workers',
        kind: 'number',
        default: 4,
        range: { min: 1, max: 16 },
        unit: 'threads',
      },
      4,
    );
    // Numeric input still present (lookup by id since the unit suffix would
    // otherwise also match a label-based query).
    const input = container.querySelector('#p-workers') as HTMLInputElement;
    expect(input).not.toBeNull();
    expect(input.type).toBe('number');
    // Slider rendered alongside, addressable by aria-label.
    const slider = screen.getByLabelText(/workers slider/i) as HTMLInputElement;
    expect(slider.type).toBe('range');
    expect(slider.min).toBe('1');
    expect(slider.max).toBe('16');
    // Unit label is rendered as a suffix.
    expect(screen.getByText('threads')).toBeInTheDocument();
  });

  test('slider drag fires onChange with the new numeric value', () => {
    const { onChange } = renderField(
      {
        name: 'workers',
        kind: 'number',
        default: 4,
        range: { min: 1, max: 16 },
      },
      4,
    );
    const slider = screen.getByLabelText(/workers slider/i) as HTMLInputElement;
    fireEvent.change(slider, { target: { value: '8' } });
    expect(onChange).toHaveBeenCalledWith(8);
  });
});

describe('ParamField - select', () => {
  test('renders a native select with options', () => {
    renderField(
      { name: 'judge', kind: 'select', options: ['gpt', 'claude'] },
      'gpt',
    );
    const sel = screen.getByLabelText(/judge/i) as HTMLSelectElement;
    expect(sel.tagName).toBe('SELECT');
    expect(sel.options.length).toBe(2);
  });

  test('change fires onChange with option value', () => {
    const { onChange } = renderField(
      { name: 'judge', kind: 'select', options: ['gpt', 'claude'] },
      'gpt',
    );
    const sel = screen.getByLabelText(/judge/i) as HTMLSelectElement;
    fireEvent.change(sel, { target: { value: 'claude' } });
    expect(onChange).toHaveBeenCalledWith('claude');
  });
});

describe('ParamField - multiselect', () => {
  test('renders a chip per option', () => {
    renderField(
      { name: 'tags', kind: 'multiselect', options: ['a', 'b', 'c'] },
      ['a'],
    );
    expect(screen.getByRole('button', { name: /a$/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /b$/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /c$/ })).toBeInTheDocument();
  });

  test('clicking off chip adds it to the array', () => {
    const { onChange } = renderField(
      { name: 'tags', kind: 'multiselect', options: ['a', 'b'] },
      ['a'],
    );
    fireEvent.click(screen.getByRole('button', { name: /b$/ }));
    expect(onChange).toHaveBeenCalledWith(['a', 'b']);
  });

  test('clicking on chip removes it from the array', () => {
    const { onChange } = renderField(
      { name: 'tags', kind: 'multiselect', options: ['a', 'b'] },
      ['a', 'b'],
    );
    fireEvent.click(screen.getByRole('button', { name: /a$/ }));
    expect(onChange).toHaveBeenCalledWith(['b']);
  });
});

describe('ParamField - path', () => {
  test('renders an input with path placeholder', () => {
    renderField({ name: 'dataset', kind: 'path' }, '');
    const input = screen.getByLabelText(/dataset/i) as HTMLInputElement;
    expect(input.placeholder).toBe('path/to/file or directory');
  });
});

describe('ParamField - long-text', () => {
  test('renders a textarea', () => {
    renderField({ name: 'note', kind: 'long-text' }, 'hello');
    const ta = screen.getByLabelText(/note/i) as HTMLTextAreaElement;
    expect(ta.tagName).toBe('TEXTAREA');
    expect(ta.value).toBe('hello');
  });
});

describe('ParamField - markers', () => {
  test('required renders an asterisk', () => {
    renderField({ name: 'name', kind: 'string', required: true });
    expect(screen.getByText('*')).toBeInTheDocument();
  });

  test('advanced renders an "advanced" tag', () => {
    renderField({ name: 'name', kind: 'string', advanced: true });
    expect(screen.getByText('advanced')).toBeInTheDocument();
  });

  test('help text renders below the control', () => {
    renderField({ name: 'name', kind: 'string', help: 'a thing' });
    expect(screen.getByText('a thing')).toBeInTheDocument();
  });
});
