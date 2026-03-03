/**
 * F0 Canvas Editor - Vanilla JS port of gradio_f0_editor canvas.ts + history.ts
 *
 * Provides interactive F0/Energy curve editing via HTML Canvas.
 * Usage: new F0CanvasEditor(canvasElement)
 */

// ============================================================================
// EditHistory
// ============================================================================

class EditHistory {
  constructor(maxSize = 100) {
    this._stack = [];
    this._pointer = -1;
    this._maxSize = maxSize;
  }

  push(snapshot) {
    // Discard any redo states
    this._stack = this._stack.slice(0, this._pointer + 1);
    // Deep copy
    const copy = {
      f0: snapshot.f0.map((a) => [...a]),
      n: snapshot.n.map((a) => [...a]),
    };
    this._stack.push(copy);
    if (this._stack.length > this._maxSize) {
      this._stack.shift();
    } else {
      this._pointer++;
    }
  }

  undo() {
    if (this._pointer <= 0) return null;
    this._pointer--;
    return this.getCurrent();
  }

  redo() {
    if (this._pointer >= this._stack.length - 1) return null;
    this._pointer++;
    return this.getCurrent();
  }

  getCurrent() {
    if (this._pointer < 0 || this._pointer >= this._stack.length) return null;
    const s = this._stack[this._pointer];
    return {
      f0: s.f0.map((a) => [...a]),
      n: s.n.map((a) => [...a]),
    };
  }

  get canUndo() {
    return this._pointer > 0;
  }

  get canRedo() {
    return this._pointer < this._stack.length - 1;
  }

  clear() {
    this._stack = [];
    this._pointer = -1;
  }
}

// ============================================================================
// F0CanvasEditor
// ============================================================================

const COLORS = {
  f0: "#1f77b4",
  f0_original: "rgba(150,150,150,0.45)",
  energy: "#ff7f0e",
  energy_original: "rgba(200,180,150,0.45)",
  phonemeBorder: "rgba(100,100,100,0.3)",
  phonemeLabel: "rgba(80,80,80,0.7)",
  sentenceBorder: "rgba(200,50,50,0.5)",
  selection: "rgba(100,150,255,0.15)",
  selectionBorder: "rgba(100,150,255,0.5)",
  grid: "rgba(0,0,0,0.06)",
  gridLabel: "rgba(0,0,0,0.35)",
  bg: "#fafafa",
};

const PADDING = { top: 20, right: 20, bottom: 35, left: 55 };

class F0CanvasEditor {
  constructor(canvas) {
    this._canvas = canvas;
    this._ctx = canvas.getContext("2d");
    this._data = null;
    this._history = new EditHistory();

    // View state
    this._view = { offsetX: 0, scaleX: 1, scaleY: 1, offsetY: 0 };

    // Edit state
    this._editTarget = "f0";
    this._isDragging = false;
    this._isPanning = false;
    this._dragStartX = 0;
    this._dragStartY = 0;
    this._lastMouseFrame = -1;
    this._lastMouseY = 0;
    this._selStart = null;
    this._selEnd = null;
    this._hasEditedSinceSnapshot = false;

    // Callbacks
    this._onChange = null;

    // Bound handlers (for removal)
    this._boundMouseDown = this._onMouseDown.bind(this);
    this._boundMouseMove = this._onMouseMove.bind(this);
    this._boundMouseUp = this._onMouseUp.bind(this);
    this._boundWheel = this._onWheel.bind(this);
    this._boundContextMenu = (e) => e.preventDefault();

    this._attachEvents();
  }

  // --- Public API ---

  setData(data) {
    this._data = data;
    this._history.clear();
    if (data) {
      this._pushSnapshot();
      this._fitView();
    }
    this.render();
  }

  getData() {
    return this._data;
  }

  setEditTarget(target) {
    this._editTarget = target;
    this.render();
  }

  getEditTarget() {
    return this._editTarget;
  }

  toggleTarget() {
    this._editTarget = this._editTarget === "f0" ? "energy" : "f0";
    this.render();
    return this._editTarget;
  }

  setOnChange(cb) {
    this._onChange = cb;
  }

  get canUndo() {
    return this._history.canUndo;
  }

  get canRedo() {
    return this._history.canRedo;
  }

  undo() {
    const snap = this._history.undo();
    if (snap && this._data) {
      this._applySnapshot(snap);
      this.render();
      this._emitChange();
    }
  }

  redo() {
    const snap = this._history.redo();
    if (snap && this._data) {
      this._applySnapshot(snap);
      this.render();
      this._emitChange();
    }
  }

  resetToOriginal() {
    if (!this._data) return;
    for (const s of this._data.sentences) {
      s.f0 = [...s.f0_original];
      s.n = [...s.n_original];
    }
    this._pushSnapshot();
    this.render();
    this._emitChange();
  }

  clearSelection() {
    this._selStart = null;
    this._selEnd = null;
    this.render();
  }

  resize(width, height) {
    const dpr = window.devicePixelRatio || 1;
    this._canvas.width = width * dpr;
    this._canvas.height = height * dpr;
    this._canvas.style.width = `${width}px`;
    this._canvas.style.height = `${height}px`;
    this._ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.render();
  }

  destroy() {
    this._canvas.removeEventListener("mousedown", this._boundMouseDown);
    this._canvas.removeEventListener("mousemove", this._boundMouseMove);
    this._canvas.removeEventListener("mouseup", this._boundMouseUp);
    this._canvas.removeEventListener("mouseleave", this._boundMouseUp);
    this._canvas.removeEventListener("wheel", this._boundWheel);
    this._canvas.removeEventListener("contextmenu", this._boundContextMenu);
  }

  // --- Coordinate transforms ---

  get _plotW() {
    return (
      this._canvas.width / (window.devicePixelRatio || 1) -
      PADDING.left -
      PADDING.right
    );
  }

  get _plotH() {
    return (
      this._canvas.height / (window.devicePixelRatio || 1) -
      PADDING.top -
      PADDING.bottom
    );
  }

  get _totalFrames() {
    if (!this._data) return 0;
    return this._data.sentences.reduce((sum, s) => sum + s.f0.length, 0);
  }

  _frameToX(frame) {
    const framesVisible = this._totalFrames / this._view.scaleX;
    return (
      PADDING.left +
      ((frame - this._view.offsetX) / framesVisible) * this._plotW
    );
  }

  _xToFrame(x) {
    const framesVisible = this._totalFrames / this._view.scaleX;
    return (
      ((x - PADDING.left) / this._plotW) * framesVisible + this._view.offsetX
    );
  }

  _valueToY(val, minVal, maxVal) {
    const range = maxVal - minVal || 1;
    const adjusted = val - this._view.offsetY;
    const norm = (adjusted - minVal) / range;
    return PADDING.top + this._plotH * (1 - norm);
  }

  _yToValue(y, minVal, maxVal) {
    const range = maxVal - minVal || 1;
    const norm = 1 - (y - PADDING.top) / this._plotH;
    return norm * range + minVal + this._view.offsetY;
  }

  // --- Data range ---

  _getValueRange() {
    if (!this._data) return { min: 0, max: 1 };
    const arr = this._editTarget === "f0" ? "f0" : "n";
    const origArr = this._editTarget === "f0" ? "f0_original" : "n_original";
    let min = Infinity,
      max = -Infinity;
    for (const s of this._data.sentences) {
      for (const v of s[arr]) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
      for (const v of s[origArr]) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    if (!isFinite(min)) {
      min = 0;
      max = 1;
    }
    const pad = (max - min) * 0.1 || 0.5;
    return { min: min - pad, max: max + pad };
  }

  _fitView() {
    this._view = { offsetX: 0, scaleX: 1, scaleY: 1, offsetY: 0 };
  }

  // --- Rendering ---

  render() {
    const w = this._canvas.width / (window.devicePixelRatio || 1);
    const h = this._canvas.height / (window.devicePixelRatio || 1);
    const ctx = this._ctx;

    // Background
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, w, h);

    if (!this._data || this._data.sentences.length === 0) {
      ctx.fillStyle = "#999";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(
        "\u30D7\u30EC\u30D3\u30E5\u30FC\u3092\u5B9F\u884C\u3057\u3066F0\u30C7\u30FC\u30BF\u3092\u8868\u793A",
        w / 2,
        h / 2
      );
      return;
    }

    const { min: valMin, max: valMax } = this._getValueRange();

    this._drawGrid(valMin, valMax);
    this._drawSelection();
    this._drawPhonemes(valMin, valMax);
    this._drawSentenceBorders();
    this._drawCurves(valMin, valMax);
    this._drawAxes(valMin, valMax);
  }

  _drawGrid(valMin, valMax) {
    const ctx = this._ctx;
    if (!this._data) return;

    // Horizontal grid lines (value)
    const range = valMax - valMin;
    const step = this._niceStep(range, 6);
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    ctx.setLineDash([]);
    for (let v = Math.ceil(valMin / step) * step; v <= valMax; v += step) {
      const y = this._valueToY(v, valMin, valMax);
      ctx.beginPath();
      ctx.moveTo(PADDING.left, y);
      ctx.lineTo(PADDING.left + this._plotW, y);
      ctx.stroke();
    }

    // Vertical grid lines (time)
    const sr = this._data.sr;
    const hop = this._data.hop_length;
    const framesVisible = this._totalFrames / this._view.scaleX;
    const totalSec = (framesVisible * hop) / sr;
    const timeStep = this._niceStep(totalSec, 8);

    for (let t = 0; t <= totalSec + timeStep; t += timeStep) {
      const frame = (t * sr) / hop + this._view.offsetX;
      const x = this._frameToX(frame);
      if (x < PADDING.left || x > PADDING.left + this._plotW) continue;
      ctx.beginPath();
      ctx.moveTo(x, PADDING.top);
      ctx.lineTo(x, PADDING.top + this._plotH);
      ctx.stroke();
    }
  }

  _drawAxes(valMin, valMax) {
    const ctx = this._ctx;
    if (!this._data) return;

    const sr = this._data.sr;
    const hop = this._data.hop_length;

    // Y axis labels
    const range = valMax - valMin;
    const step = this._niceStep(range, 6);
    ctx.fillStyle = COLORS.gridLabel;
    ctx.font = "11px sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let v = Math.ceil(valMin / step) * step; v <= valMax; v += step) {
      const y = this._valueToY(v, valMin, valMax);
      ctx.fillText(v.toFixed(1), PADDING.left - 5, y);
    }

    // X axis labels (time)
    const framesVisible = this._totalFrames / this._view.scaleX;
    const totalSec = (framesVisible * hop) / sr;
    const timeStep = this._niceStep(totalSec, 8);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let t = 0; t <= totalSec + timeStep; t += timeStep) {
      const frame = (t * sr) / hop + this._view.offsetX;
      const x = this._frameToX(frame);
      if (x < PADDING.left || x > PADDING.left + this._plotW) continue;
      ctx.fillText(`${t.toFixed(2)}s`, x, PADDING.top + this._plotH + 5);
    }

    // Axis label
    ctx.save();
    ctx.translate(12, PADDING.top + this._plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(this._editTarget === "f0" ? "F0" : "Energy", 0, 0);
    ctx.restore();
  }

  _drawCurves(valMin, valMax) {
    const ctx = this._ctx;
    if (!this._data) return;

    let frameOff = 0;
    const activeArr = this._editTarget === "f0" ? "f0" : "n";
    const origArr = this._editTarget === "f0" ? "f0_original" : "n_original";
    const activeColor = this._editTarget === "f0" ? COLORS.f0 : COLORS.energy;
    const origColor =
      this._editTarget === "f0" ? COLORS.f0_original : COLORS.energy_original;

    for (const s of this._data.sentences) {
      const arr = s[activeArr];
      const orig = s[origArr];
      const n = arr.length;

      // Original (dashed)
      ctx.strokeStyle = origColor;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = this._frameToX(frameOff + i);
        const y = this._valueToY(orig[i], valMin, valMax);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Current (solid)
      ctx.strokeStyle = activeColor;
      ctx.lineWidth = 2;
      ctx.setLineDash([]);
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = this._frameToX(frameOff + i);
        const y = this._valueToY(arr[i], valMin, valMax);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      frameOff += n;
    }
  }

  _drawPhonemes(valMin, valMax) {
    const ctx = this._ctx;
    if (!this._data) return;

    let frameOff = 0;
    ctx.strokeStyle = COLORS.phonemeBorder;
    ctx.fillStyle = COLORS.phonemeLabel;
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.font = "9px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";

    for (const s of this._data.sentences) {
      if (!s.phonemes) {
        frameOff += s.f0.length;
        continue;
      }
      for (const ph of s.phonemes) {
        const x = this._frameToX(frameOff + ph.start_frame);
        if (x >= PADDING.left && x <= PADDING.left + this._plotW) {
          ctx.beginPath();
          ctx.moveTo(x, PADDING.top);
          ctx.lineTo(x, PADDING.top + this._plotH);
          ctx.stroke();
        }
        // Label at midpoint
        const midFrame = frameOff + (ph.start_frame + ph.end_frame) / 2;
        const mx = this._frameToX(midFrame);
        if (mx >= PADDING.left && mx <= PADDING.left + this._plotW) {
          ctx.fillText(ph.label, mx, PADDING.top + 2);
        }
      }
      frameOff += s.f0.length;
    }
    ctx.setLineDash([]);
  }

  _drawSentenceBorders() {
    const ctx = this._ctx;
    if (!this._data || this._data.sentences.length <= 1) return;

    let frameOff = 0;
    ctx.strokeStyle = COLORS.sentenceBorder;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);

    for (let i = 0; i < this._data.sentences.length - 1; i++) {
      frameOff += this._data.sentences[i].f0.length;
      const x = this._frameToX(frameOff);
      if (x >= PADDING.left && x <= PADDING.left + this._plotW) {
        ctx.beginPath();
        ctx.moveTo(x, PADDING.top);
        ctx.lineTo(x, PADDING.top + this._plotH);
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);
  }

  _drawSelection() {
    if (this._selStart === null || this._selEnd === null) return;
    const ctx = this._ctx;
    const x1 = this._frameToX(Math.min(this._selStart, this._selEnd));
    const x2 = this._frameToX(Math.max(this._selStart, this._selEnd));

    ctx.fillStyle = COLORS.selection;
    ctx.fillRect(x1, PADDING.top, x2 - x1, this._plotH);
    ctx.strokeStyle = COLORS.selectionBorder;
    ctx.lineWidth = 1;
    ctx.setLineDash([]);
    ctx.strokeRect(x1, PADDING.top, x2 - x1, this._plotH);
  }

  // --- Event handling ---

  _attachEvents() {
    this._canvas.addEventListener("mousedown", this._boundMouseDown);
    this._canvas.addEventListener("mousemove", this._boundMouseMove);
    this._canvas.addEventListener("mouseup", this._boundMouseUp);
    this._canvas.addEventListener("mouseleave", this._boundMouseUp);
    this._canvas.addEventListener("wheel", this._boundWheel, {
      passive: false,
    });
    this._canvas.addEventListener("contextmenu", this._boundContextMenu);
  }

  _onMouseDown(e) {
    if (!this._data) return;
    const rect = this._canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Right-click or Shift+click: pan
    if (e.button === 2 || e.shiftKey) {
      this._isPanning = true;
      this._dragStartX = x;
      this._dragStartY = y;
      return;
    }

    const frame = Math.round(this._xToFrame(x));
    this._dragStartX = x;
    this._dragStartY = y;

    // Alt+click: range select
    if (e.altKey) {
      this._selStart = frame;
      this._selEnd = frame;
      this._isDragging = false;
      return;
    }

    // Normal click: start drawing/editing
    this._isDragging = true;
    this._hasEditedSinceSnapshot = false;
    this._lastMouseFrame = frame;
    this._lastMouseY = y;
  }

  _onMouseMove(e) {
    if (!this._data) return;
    const rect = this._canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (this._isPanning) {
      const dx = x - this._dragStartX;
      const dy = y - this._dragStartY;
      const framesPerPx = this._totalFrames / this._view.scaleX / this._plotW;
      this._view.offsetX -= dx * framesPerPx;
      this._view.offsetX = Math.max(
        0,
        Math.min(
          this._view.offsetX,
          this._totalFrames * (1 - 1 / this._view.scaleX)
        )
      );

      const { min: valMin, max: valMax } = this._getValueRange();
      const valPerPx = (valMax - valMin) / this._plotH;
      this._view.offsetY += dy * valPerPx;

      this._dragStartX = x;
      this._dragStartY = y;
      this.render();
      return;
    }

    // Range selection
    if (this._selStart !== null && e.altKey && !this._isDragging) {
      this._selEnd = Math.round(this._xToFrame(x));
      this.render();
      return;
    }

    if (!this._isDragging) return;

    // Drawing: edit F0/N along the mouse path
    const frame = Math.round(this._xToFrame(x));
    const { min: valMin, max: valMax } = this._getValueRange();
    const val = this._yToValue(y, valMin, valMax);

    if (!this._hasEditedSinceSnapshot) {
      this._hasEditedSinceSnapshot = true;
    }

    // If we have a selection, shift all selected frames
    if (this._selStart !== null && this._selEnd !== null) {
      const dy = y - this._lastMouseY;
      const valPerPx = (valMax - valMin) / this._plotH;
      const dVal = -dy * valPerPx;
      this._shiftRange(
        Math.min(this._selStart, this._selEnd),
        Math.max(this._selStart, this._selEnd),
        dVal
      );
      this._lastMouseY = y;
      this.render();
      this._emitChange();
      return;
    }

    // Point-by-point drawing
    const startF = Math.min(this._lastMouseFrame, frame);
    const endF = Math.max(this._lastMouseFrame, frame);

    for (let f = startF; f <= endF; f++) {
      this._setFrameValue(f, val);
    }

    this._lastMouseFrame = frame;
    this._lastMouseY = y;
    this.render();
    this._emitChange();
  }

  _onMouseUp(_e) {
    if (this._isPanning) {
      this._isPanning = false;
      return;
    }
    if (this._isDragging && this._hasEditedSinceSnapshot) {
      this._pushSnapshot();
    }
    this._isDragging = false;
  }

  _onWheel(e) {
    e.preventDefault();
    if (!this._data) return;

    const rect = this._canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const frame = this._xToFrame(x);
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;

    if (e.ctrlKey) {
      // Vertical zoom
      this._view.scaleY = Math.max(
        0.1,
        Math.min(10, this._view.scaleY * factor)
      );
    } else {
      // Horizontal zoom
      const newScale = Math.max(
        1,
        Math.min(50, this._view.scaleX * factor)
      );
      // Zoom toward mouse position
      this._view.scaleX = newScale;
      const newFramesVis = this._totalFrames / this._view.scaleX;
      const ratio = (x - PADDING.left) / this._plotW;
      this._view.offsetX = frame - ratio * newFramesVis;
      this._view.offsetX = Math.max(
        0,
        Math.min(this._view.offsetX, this._totalFrames - newFramesVis)
      );
    }
    this.render();
  }

  // --- Data manipulation ---

  _setFrameValue(globalFrame, value) {
    if (!this._data) return;
    const arr = this._editTarget === "f0" ? "f0" : "n";
    let off = 0;
    for (const s of this._data.sentences) {
      const n = s[arr].length;
      if (globalFrame >= off && globalFrame < off + n) {
        s[arr][globalFrame - off] = value;
        return;
      }
      off += n;
    }
  }

  _shiftRange(startFrame, endFrame, delta) {
    if (!this._data) return;
    const arr = this._editTarget === "f0" ? "f0" : "n";
    let off = 0;
    for (const s of this._data.sentences) {
      const n = s[arr].length;
      const localStart = Math.max(0, startFrame - off);
      const localEnd = Math.min(n, endFrame - off);
      for (let i = localStart; i < localEnd; i++) {
        s[arr][i] += delta;
      }
      off += n;
    }
  }

  // --- Snapshot ---

  _pushSnapshot() {
    if (!this._data) return;
    this._history.push({
      f0: this._data.sentences.map((s) => [...s.f0]),
      n: this._data.sentences.map((s) => [...s.n]),
    });
  }

  _applySnapshot(snap) {
    if (!this._data) return;
    for (let i = 0; i < this._data.sentences.length; i++) {
      if (snap.f0[i]) this._data.sentences[i].f0 = [...snap.f0[i]];
      if (snap.n[i]) this._data.sentences[i].n = [...snap.n[i]];
    }
  }

  _emitChange() {
    if (this._onChange && this._data) {
      this._onChange(this._data);
    }
  }

  // --- Helpers ---

  _niceStep(range, targetTicks) {
    const rough = range / targetTicks;
    const mag = Math.pow(10, Math.floor(Math.log10(rough)));
    const residual = rough / mag;
    let nice;
    if (residual <= 1.5) nice = 1;
    else if (residual <= 3) nice = 2;
    else if (residual <= 7) nice = 5;
    else nice = 10;
    return nice * mag;
  }
}

// Global export
window.F0CanvasEditor = F0CanvasEditor;
