// States must match Python
export const UNBURNABLE = 0;
export const UNIGNITED = 1;
export const BURNING   = 2;
export const BURNED    = 3;

export function drawCA(ctx, cells, h, w, cellSize) {
  const imgW = w * cellSize;
  const imgH = h * cellSize;
  const imageData = ctx.createImageData(imgW, imgH);
  const data = imageData.data;

  // Simple palette (you can customize)
  // unburnable: dark gray, unignited: green, burning: red/orange, burned: black
  const palette = {
    [UNBURNABLE]: [80, 80, 80, 255],
    [UNIGNITED]:  [40, 140, 70, 255],
    [BURNING]:    [230, 60, 40, 255],
    [BURNED]:     [25, 25, 25, 255],
  };

  // Paint each CA cell as a cellSize x cellSize block
  for (let i = 0; i < h; i++) {
    for (let j = 0; j < w; j++) {
      const state = cells[i * w + j];
      const [r, g, b, a] = palette[state] || [255, 0, 255, 255];

      const px0 = j * cellSize;
      const py0 = i * cellSize;
      for (let dy = 0; dy < cellSize; dy++) {
        for (let dx = 0; dx < cellSize; dx++) {
          const px = px0 + dx;
          const py = py0 + dy;
          const idx = (py * imgW + px) * 4;
          data[idx + 0] = r;
          data[idx + 1] = g;
          data[idx + 2] = b;
          data[idx + 3] = a;
        }
      }
    }
  }

  ctx.putImageData(imageData, 0, 0);
}