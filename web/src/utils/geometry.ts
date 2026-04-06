export interface Point {
  x: number;
  y: number;
}

/**
 * Ramer-Douglas-Peucker algorithm to simplify a path of points
 */
export const simplifyPath = (points: Point[], epsilon: number): Point[] => {
  if (points.length <= 2) return points;

  let dmax = 0;
  let index = 0;
  const last = points.length - 1;

  for (let i = 1; i < last; i++) {
    const d = perpendicularDistance(points[i], points[0], points[last]);
    if (d > dmax) {
      index = i;
      dmax = d;
    }
  }

  if (dmax > epsilon) {
    const res1 = simplifyPath(points.slice(0, index + 1), epsilon);
    const res2 = simplifyPath(points.slice(index), epsilon);
    return [...res1.slice(0, res1.length - 1), ...res2];
  } else {
    return [points[0], points[last]];
  }
};

const perpendicularDistance = (p: Point, p1: Point, p2: Point): number => {
  let x = p1.x;
  let y = p1.y;
  let dx = p2.x - x;
  let dy = p2.y - y;

  if (dx !== 0 || dy !== 0) {
    const t = ((p.x - x) * dx + (p.y - y) * dy) / (dx * dx + dy * dy);
    if (t > 1) {
      x = p2.x;
      y = p2.y;
    } else if (t > 0) {
      x += dx * t;
      y += dy * t;
    }
  }

  dx = p.x - x;
  dy = p.y - y;
  return Math.sqrt(dx * dx + dy * dy);
};
