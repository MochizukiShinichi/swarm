import { useEffect, useRef, useState } from 'react';

// --- GOLDEN PARITY CONSTANTS ---
const WIDTH = 800.0;
const HEIGHT = 500.0;
const DT = 1.0;
const FRICTION_AIR = 0.05;
const AGENT_RADIUS = 5.0;
const COLLISION_RADIUS = 10.0;
const AGENT_MASS = 0.05026;
const OBJ_SIZE = 40.0;
const OBJ_MASS = 6.4;

const K_AGENT = 0.02; 
const K_OBJ = 0.05;   
const K_OBS = 0.1;

interface Vector { x: number; y: number; }

// Canonical runtime policy location:
// - trainer reads/writes `web/public/policy.json`
// - Vite dev server serves it at `${BASE_URL}policy.json`
// - production builds copy that file into `dist/policy.json`
const LEGACY_PHASE2_FALLBACK_DIFFICULTY = 0.7;

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const obstacleCountForDifficulty = (diff: number) => {
    let numObs = 0;
    if (diff > 0.3) numObs = 2;
    if (diff > 0.6) numObs = 4;
    return numObs;
};

const inferDifficultyFromPolicy = (policy: any) => {
    const metaDifficulty = Number(policy?.meta?.difficulty);
    if (Number.isFinite(metaDifficulty)) {
        return clamp(metaDifficulty, 0.0, 1.0);
    }

    // Legacy phase-2 checkpoints were exported without meta.difficulty.
    // We cannot recover the true curriculum state from the file itself, so
    // use the current trained difficulty rather than silently evaluating
    // obstacle-free.
    if (Array.isArray(policy?.w1) && policy.w1.length === 22) {
        return LEGACY_PHASE2_FALLBACK_DIFFICULTY;
    }

    return 0.0;
};

const normalize = (v: Vector): Vector => {
    const mag = Math.sqrt(v.x * v.x + v.y * v.y);
    return mag > 0.001 ? { x: v.x / mag, y: v.y / mag } : { x: 0, y: 0 };
};

const sigmoid = (x: number) => 1.0 / (1.0 + Math.exp(-x));

export const usePhysics = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [brainActive, setBrainActive] = useState(false);
  const [success, setSuccess] = useState(false);
  const successRef = useRef(false);
  
  // State
  const agentsRef = useRef<{
      x: number, y: number, vx: number, vy: number, 
      h: number[], msg: number[], motorQueue: Vector[]
  }[]>([]);
  const objPosRef = useRef<{x: number, y: number, vx: number, vy: number}>({ x: 400, y: 250, vx: 0, vy: 0 });
  const obstaclesRef = useRef<{x: number, y: number, active: boolean}[]>([]);
  const targetRef = useRef<Vector | null>(null);
  const policyRef = useRef<any>(null);
  const difficultyRef = useRef(0.0);
  const trailRef = useRef<Vector[]>([]);

  const clearObstacles = () => {
      obstaclesRef.current = Array.from({ length: 4 }, () => ({ x: 0, y: 0, active: false }));
  };

  const generateObstacles = (diff: number, obj: Vector, target: Vector | null) => {
      if (!target) {
          clearObstacles();
          return;
      }

      const obs = [];
      const numObs = obstacleCountForDifficulty(diff);
      const dx = target.x - obj.x, dy = target.y - obj.y;
      const dist = Math.sqrt(dx*dx + dy*dy) + 1e-5;

      for (let i = 0; i < 4; i++) {
          if (i < numObs) {
              const t = 0.3 + 0.4 * Math.random();
              const midX = obj.x + dx * t, midY = obj.y + dy * t;
              const px = -dy/dist, py = dx/dist;
              obs.push({ 
                  x: midX + px * (Math.random()-0.5) * 150.0, 
                  y: midY + py * (Math.random()-0.5) * 150.0, 
                  active: true 
              });
          } else {
              obs.push({ x: 0, y: 0, active: false });
          }
      }
      obstaclesRef.current = obs;
  };

  const parkAgents = () => {
      const obj = objPosRef.current;
      const cols = 10;
      const spacing = 12;
      const startX = obj.x - ((cols - 1) * spacing) / 2;
      const startY = obj.y + 70;

      agentsRef.current.forEach((a, i) => {
          const row = Math.floor(i / cols);
          const col = i % cols;
          a.x = startX + col * spacing;
          a.y = startY + row * spacing;
          a.vx = 0;
          a.vy = 0;
          a.h = new Array(16).fill(0);
          a.msg = new Array(4).fill(0);
          a.motorQueue = [{x:0, y:0}, {x:0, y:0}];
      });
  };

  // Load Policy from public/policy.json (Live parity)
  useEffect(() => {
    const loadPolicy = async () => {
        try {
            const res = await fetch(`${import.meta.env.BASE_URL}policy.json?t=${Date.now()}`);
            const data = await res.json();
            policyRef.current = data;
            difficultyRef.current = inferDifficultyFromPolicy(data);
            if (targetRef.current) {
                generateObstacles(difficultyRef.current, objPosRef.current, targetRef.current);
            } else {
                clearObstacles();
            }
            setBrainActive(true);
            if (data?.meta?.difficulty === undefined) {
                console.warn(`Policy loaded without meta.difficulty; using fallback difficulty ${difficultyRef.current.toFixed(2)} for web evaluation.`);
            } else {
                console.log("Policy Loaded. Difficulty:", difficultyRef.current);
            }
        } catch (err) {
            console.error("Failed to fetch policy data.", err);
        }
    };
    loadPolicy();
    const interval = setInterval(loadPolicy, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const initialAgents = [];
    for (let i = 0; i < 100; i++) {
        initialAgents.push({
            x: 0, y: 0,
            vx: 0, vy: 0,
            h: new Array(16).fill(0),
            msg: new Array(4).fill(0),
            motorQueue: [{x:0, y:0}, {x:0, y:0}]
        });
    }
    agentsRef.current = initialAgents;
    parkAgents();
    clearObstacles();

    let animationId: number;
    const loop = () => {
        const ctx = canvasRef.current?.getContext('2d');
        if (!ctx) {
            animationId = requestAnimationFrame(loop);
            return;
        }

        const obj = objPosRef.current;
        const target = targetRef.current;
        const policy = policyRef.current;
        const obstacles = obstaclesRef.current;
        const diff = difficultyRef.current;

        if (policy && target && !successRef.current) {
            // Local Consensus
            const gridVX = new Float32Array(40 * 40);
            const gridVY = new Float32Array(40 * 40);
            const gridMsg = new Float32Array(40 * 40 * 4);
            const gridCount = new Int32Array(40 * 40);
            
            agentsRef.current.forEach(a => {
                const gx = Math.max(0, Math.min(39, Math.floor(a.x / 20)));
                const gy = Math.max(0, Math.min(39, Math.floor(a.y / 20)));
                const idx = gx * 40 + gy;
                gridVX[idx] += a.vx;
                gridVY[idx] += a.vy;
                gridCount[idx]++;
                for (let k = 0; k < 4; k++) gridMsg[idx * 4 + k] += a.msg[k];
            });

            // Trail
            if (target) {
                trailRef.current.push({ x: obj.x, y: obj.y });
                if (trailRef.current.length > 300) trailRef.current.shift();
            }

            agentsRef.current.forEach((agent) => {
                const gx = Math.max(0, Math.min(39, Math.floor(agent.x / 20)));
                const gy = Math.max(0, Math.min(39, Math.floor(agent.y / 20)));
                let sumVX = 0, sumVY = 0, count = 0;
                let sumMsg = [0, 0, 0, 0];
                
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        const nx = gx + dx, ny = gy + dy;
                        if (nx >= 0 && nx < 40 && ny >= 0 && ny < 40) {
                            const idx = nx * 40 + ny;
                            sumVX += gridVX[idx];
                            sumVY += gridVY[idx];
                            count += gridCount[idx];
                            for (let k = 0; k < 4; k++) sumMsg[k] += gridMsg[idx * 4 + k];
                        }
                    }
                }
                const avgVX = count > 0 ? sumVX / count : 0;
                const avgVY = count > 0 ? sumVY / count : 0;
                const avgMsg = sumMsg.map(m => count > 0 ? m / count : 0);

                // Sensors
                const toT = target ? normalize({ x: target.x - obj.x, y: target.y - obj.y }) : { x: 0, y: 0 };
                const dO = { x: obj.x - agent.x, y: obj.y - agent.y };
                const distO = Math.sqrt(dO.x * dO.x + dO.y * dO.y);
                const toO = normalize(dO);
                const rp = { x: (agent.x - obj.x) / 100.0, y: (agent.y - obj.y) / 100.0 };
                const wL = 1.0 / (1.0 + agent.x), wR = 1.0 / (1.0 + (WIDTH - agent.x));
                const wT = 1.0 / (1.0 + agent.y), wB = 1.0 / (1.0 + (HEIGHT - agent.y));

                const inp = [
                    toT.x, toT.y, toO.x, toO.y, rp.x, rp.y, agent.vx, agent.vy, avgVX, avgVY,
                    wL, wR, wT, wB, 
                    distO / 100.0, obj.vx, obj.vy, Math.sqrt(agent.vx*agent.vx + agent.vy*agent.vy),
                    avgMsg[0], avgMsg[1], avgMsg[2], avgMsg[3]
                ];

                // Inference
                const h_mlp1 = [];
                for (let j = 0; j < 64; j++) {
                    let val = policy.b1[j];
                    for (let k = 0; k < 22; k++) val += inp[k] * policy.w1[k][j];
                    h_mlp1.push(Math.tanh(val));
                }
                const z_gate = [], r_gate = [];
                for (let j = 0; j < 16; j++) {
                    let vz = policy.b_gru[0][j], vr = policy.b_gru[1][j];
                    for (let k = 0; k < 64; k++) {
                        vz += h_mlp1[k] * policy.w_gru_x[0][k][j];
                        vr += h_mlp1[k] * policy.w_gru_x[1][k][j];
                    }
                    for (let k = 0; k < 16; k++) {
                        vz += agent.h[k] * policy.w_gru_h[0][k][j];
                        vr += agent.h[k] * policy.w_gru_h[1][k][j];
                    }
                    z_gate.push(sigmoid(vz)); r_gate.push(sigmoid(vr));
                }
                const h_hat = [];
                for (let j = 0; j < 16; j++) {
                    let vh = policy.b_gru[2][j];
                    for (let k = 0; k < 64; k++) vh += h_mlp1[k] * policy.w_gru_x[2][k][j];
                    for (let k = 0; k < 16; k++) vh += r_gate[j] * agent.h[k] * policy.w_gru_h[2][k][j];
                    h_hat.push(Math.tanh(vh));
                }
                for (let j = 0; j < 16; j++) {
                    const nextH = (1.0 - z_gate[j]) * agent.h[j] + z_gate[j] * h_hat[j];
                    agent.h[j] = isNaN(nextH) ? 0 : nextH;
                }

                const out = [];
                for (let j = 0; j < 6; j++) {
                    let val = policy.b2[j];
                    for (let k = 0; k < 16; k++) val += agent.h[k] * policy.w2[k][j];
                    out.push(Math.tanh(val));
                }

                const fCurrent = { x: out[0] * 0.02, y: out[1] * 0.02 };
                let fDelayed = fCurrent;
                if (diff > 0.6) {
                    fDelayed = agent.motorQueue[1];
                    agent.motorQueue[1] = agent.motorQueue[0];
                    agent.motorQueue[0] = fCurrent;
                }
                agent.vx += (fDelayed.x / AGENT_MASS) * DT;
                agent.vy += (fDelayed.y / AGENT_MASS) * DT;
                for (let k = 0; k < 4; k++) agent.msg[k] = out[2+k];
            });

            // Physics
            for (let i = 0; i < 100; i++) {
                const a = agentsRef.current[i];
                for (let j = i + 1; j < 100; j++) {
                    const b = agentsRef.current[j];
                    const dx = a.x - b.x, dy = a.y - b.y;
                    const d = Math.sqrt(dx*dx + dy*dy) + 1e-5;
                    if (d < COLLISION_RADIUS) {
                        const pushF = (COLLISION_RADIUS - d) * K_AGENT;
                        const fx = (dx/d) * pushF, fy = (dy/d) * pushF;
                        a.vx += (fx / AGENT_MASS) * DT; a.vy += (fy / AGENT_MASS) * DT;
                        b.vx -= (fx / AGENT_MASS) * DT; b.vy -= (fy / AGENT_MASS) * DT;
                    }
                }
                const odx = a.x - obj.x, ody = a.y - obj.y;
                if (Math.abs(odx) < OBJ_SIZE + AGENT_RADIUS && Math.abs(ody) < OBJ_SIZE + AGENT_RADIUS) {
                    const px = OBJ_SIZE + AGENT_RADIUS - Math.abs(odx), py = OBJ_SIZE + AGENT_RADIUS - Math.abs(ody);
                    let fx = 0, fy = 0;
                    if (px < py) fx = (odx >= 0 ? 1 : -1) * px * K_OBJ; else fy = (ody >= 0 ? 1 : -1) * py * K_OBJ;
                    a.vx += (fx / AGENT_MASS) * DT; a.vy += (fy / AGENT_MASS) * DT;
                    obj.vx -= (fx / OBJ_MASS) * DT; obj.vy -= (fy / OBJ_MASS) * DT;
                }
                obstacles.forEach(obs => {
                    if (obs.active) {
                        const dx = a.x - obs.x, dy = a.y - obs.y;
                        const d = Math.sqrt(dx*dx + dy*dy) + 1e-5;
                        const rSum = AGENT_RADIUS + 20.0;
                        if (d < rSum) {
                            const pushF = (rSum - d) * K_OBS;
                            a.vx += (dx/d * pushF / AGENT_MASS) * DT; a.vy += (dy/d * pushF / AGENT_MASS) * DT;
                        }
                    }
                });
                a.vx *= (1.0 - FRICTION_AIR); a.vy *= (1.0 - FRICTION_AIR);
                a.x += a.vx * DT; a.y += a.vy * DT;
                if (a.x < 0) { a.x = 0; a.vx *= -0.5; } if (a.x > WIDTH) { a.x = WIDTH; a.vx *= -0.5; }
                if (a.y < 0) { a.y = 0; a.vy *= -0.5; } if (a.y > HEIGHT) { a.y = HEIGHT; a.vy *= -0.5; }
            }
            obj.vx *= (1.0 - FRICTION_AIR); obj.vy *= (1.0 - FRICTION_AIR);
            obstacles.forEach(obs => {
                if (obs.active) {
                    const dx = obj.x - obs.x, dy = obj.y - obs.y;
                    const d = Math.sqrt(dx*dx + dy*dy) + 1e-5;
                    const rSum = OBJ_SIZE + 20.0;
                    if (d < rSum) {
                        const pushF = (rSum - d) * K_OBS;
                        obj.vx += (dx/d * pushF / OBJ_MASS) * DT; obj.vy += (dy/d * pushF / OBJ_MASS) * DT;
                    }
                }
            });
            obj.x += obj.vx * DT; obj.y += obj.vy * DT;
            if (obj.x < OBJ_SIZE) { obj.x = OBJ_SIZE; obj.vx *= -0.5; } if (obj.x > WIDTH - OBJ_SIZE) { obj.x = WIDTH - OBJ_SIZE; obj.vx *= -0.5; }
            if (obj.y < OBJ_SIZE) { obj.y = OBJ_SIZE; obj.vy *= -0.5; } if (obj.y > HEIGHT - OBJ_SIZE) { obj.y = HEIGHT - OBJ_SIZE; obj.vy *= -0.5; }
            if (target && Math.sqrt((obj.x - target.x)**2 + (obj.y - target.y)**2) < 40.0) { successRef.current = true; setSuccess(true); }
        }

        // RENDER
        ctx.clearRect(0, 0, WIDTH, HEIGHT);
        ctx.setLineDash([]);
        obstacles.forEach(obs => {
            if (obs.active) {
                ctx.fillStyle = '#475569'; ctx.beginPath(); ctx.arc(obs.x, obs.y, 20, 0, Math.PI*2); ctx.fill();
                ctx.strokeStyle = '#64748b'; ctx.lineWidth = 2; ctx.stroke();
            }
        });
        if (trailRef.current.length > 1) {
            ctx.beginPath(); ctx.strokeStyle = '#fdba74'; ctx.lineWidth = 1; ctx.setLineDash([5, 5]);
            ctx.moveTo(trailRef.current[0].x, trailRef.current[0].y);
            trailRef.current.forEach(p => ctx.lineTo(p.x, p.y)); ctx.stroke(); ctx.setLineDash([]);
        }
        if (target) {
            ctx.fillStyle = successRef.current ? '#22c55e' : '#f97316';
            ctx.beginPath(); ctx.arc(target.x, target.y, 12, 0, Math.PI*2); ctx.fill();
        }
        ctx.strokeStyle = successRef.current ? '#22c55e' : '#fb923c'; ctx.lineWidth = 3;
        ctx.strokeRect(obj.x - OBJ_SIZE, obj.y - OBJ_SIZE, OBJ_SIZE*2, OBJ_SIZE*2);
        ctx.strokeStyle = 'rgba(34, 211, 238, 0.1)'; ctx.lineWidth = 0.5;
        agentsRef.current.forEach((a, i) => {
            if (i % 5 === 0) {
                agentsRef.current.forEach((b, j) => {
                    if (i < j && j % 10 === 0) {
                        const d = Math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2);
                        if (d < 60) { ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
                    }
                });
            }
        });
        ctx.fillStyle = successRef.current ? '#94a3b8' : '#22d3ee';
        agentsRef.current.forEach(a => { ctx.beginPath(); ctx.arc(a.x, a.y, AGENT_RADIUS, 0, Math.PI*2); ctx.fill(); });
        
        animationId = requestAnimationFrame(loop);
    };
    animationId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animationId);
  }, []);

  const setTarget = (nx: number, ny: number) => { 
      targetRef.current = { x: nx, y: ny }; 
      successRef.current = false; setSuccess(false); 
      trailRef.current = [];

      const obj = objPosRef.current;
      const diff = difficultyRef.current;

      // Align agent spawn side with engine logic
      const dx_real = nx - obj.x, dy_real = ny - obj.y;
      const dist_real = Math.sqrt(dx_real*dx_real + dy_real*dy_real) + 1e-5;
      const scX = obj.x - (dx_real / dist_real) * 60.0;
      const scY = obj.y - (dy_real / dist_real) * 60.0;

      agentsRef.current.forEach(a => {
          a.x = scX + (Math.random()-0.5) * 80.0;
          a.y = scY + (Math.random()-0.5) * 80.0;
          a.vx = 0; a.vy = 0;
          a.h = new Array(16).fill(0); a.msg = new Array(4).fill(0);
          a.motorQueue = [{x:0, y:0}, {x:0, y:0}];
      });

      generateObstacles(diff, obj, targetRef.current);
  };

  const resetEnv = () => {
      objPosRef.current = { x: 400, y: 250, vx: 0, vy: 0 };
      successRef.current = false; setSuccess(false);
      targetRef.current = null; trailRef.current = [];
      clearObstacles();
      parkAgents();
  };

  return { canvasRef, setTarget, resetEnv, agentCount: 100, brainActive, success };
};
