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
      h: number[], msg: number[], motorQueue: Vector[],
      roleColor: string
  }[]>([]);
  const objPosRef = useRef<{x: number, y: number, vx: number, vy: number}>({ x: 400, y: 250, vx: 0, vy: 0 });
  const obstaclesRef = useRef<{x: number, y: number, active: boolean}[]>([]);
  const targetRef = useRef<Vector | null>(null);
  const policyRef = useRef<any>(null);
  const trailRef = useRef<Vector[]>([]);

  // Load Policy
  useEffect(() => {
    fetch('policy.json')
      .then(r => r.json())
      .then(data => {
          policyRef.current = data;
          setBrainActive(true);
          console.log("Phase 3 Policy (Attention) Loaded");
      })
      .catch(err => console.error("Failed to load policy.json", err));
  }, []);

  useEffect(() => {
    // 1. Init
    const initialAgents = [];
    for (let i = 0; i < 100; i++) {
        const angle = Math.random() * 2.0 * Math.PI;
        const r = 60.0 + Math.random() * 40.0;
        initialAgents.push({
            x: 400 + Math.cos(angle) * r, y: 250 + Math.sin(angle) * r,
            vx: 0, vy: 0,
            h: new Array(16).fill(0),
            msg: new Array(4).fill(0),
            motorQueue: [{x:0, y:0}, {x:0, y:0}],
            roleColor: '#22d3ee'
        });
    }
    agentsRef.current = initialAgents;
    obstaclesRef.current = [];

    // 2. Loop
    let animationId: number;
    const loop = () => {
        const ctx = canvasRef.current?.getContext('2d');
        if (!ctx) { animationId = requestAnimationFrame(loop); return; }

        const obj = objPosRef.current;
        const target = targetRef.current;
        const policy = policyRef.current;
        const obstacles = obstaclesRef.current;

        if (target && policy && !successRef.current) {
            // Local Consensus (Grid Parity)
            const gridVX = new Float32Array(40 * 40);
            const gridVY = new Float32Array(40 * 40);
            const gridK = new Float32Array(40 * 40 * 4);
            const gridV = new Float32Array(40 * 40 * 4);
            const gridCount = new Int32Array(40 * 40);
            const gridAgentHead = new Int32Array(40 * 40).fill(-1);
            const gridAgentNext = new Int32Array(100);
            
            agentsRef.current.forEach((a, i) => {
                const gx = Math.max(0, Math.min(39, Math.floor(a.x / 20)));
                const gy = Math.max(0, Math.min(39, Math.floor(a.y / 20)));
                const idx = gx * 40 + gy;
                
                gridVX[idx] += a.vx; gridVY[idx] += a.vy;
                gridCount[idx]++;
                
                // Broadcast K, V (Stage 3.0)
                for (let k = 0; k < 4; k++) {
                    let kv = 0, vv = 0;
                    for (let m = 0; m < 4; m++) {
                        kv += a.msg[m] * policy.w_attn_k[m][k];
                        vv += a.msg[m] * policy.w_attn_v[m][k];
                    }
                    gridK[idx * 4 + k] += kv;
                    gridV[idx * 4 + k] += vv;
                }
                
                // Linked List for Broadphase
                gridAgentNext[i] = gridAgentHead[idx];
                gridAgentHead[idx] = i;
            });

            trailRef.current.push({ x: obj.x, y: obj.y });
            if (trailRef.current.length > 300) trailRef.current.shift();

            agentsRef.current.forEach((agent) => {
                const gx = Math.max(0, Math.min(39, Math.floor(agent.x / 20)));
                const gy = Math.max(0, Math.min(39, Math.floor(agent.y / 20)));
                let sumVX = 0, sumVY = 0, count = 0;
                let sumK = [0,0,0,0], sumV = [0,0,0,0];
                
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        const nx = gx + dx, ny = gy + dy;
                        if (nx >= 0 && nx < 40 && ny >= 0 && ny < 40) {
                            const idx = nx * 40 + ny;
                            sumVX += gridVX[idx]; sumVY += gridVY[idx];
                            count += gridCount[idx];
                            for (let k = 0; k < 4; k++) {
                                sumK[k] += gridK[idx * 4 + k];
                                sumV[k] += gridV[idx * 4 + k];
                            }
                        }
                    }
                }
                const avgVX = count > 0 ? sumVX / count : 0;
                const avgVY = count > 0 ? sumVY / count : 0;
                
                // Attention Context (Stage 3.0)
                let query = [0,0,0,0];
                for (let j = 0; j < 4; j++) {
                    let val = 0;
                    for (let k = 0; k < 16; k++) val += agent.h[k] * policy.w_attn_q[k][j];
                    query[j] = Math.tanh(val);
                }
                let score = 0;
                for (let k = 0; k < 4; k++) score += query[k] * (sumK[k] / Math.max(1, count));
                const weight = sigmoid(score);
                const attnCtx = sumV.map(v => weight * (v / Math.max(1, count)));

                // Sensors
                const toT = normalize({ x: target.x - obj.x, y: target.y - obj.y });
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
                    attnCtx[0], attnCtx[1], attnCtx[2], attnCtx[3]
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
                for (let j = 0; j < 16; j++) agent.h[j] = (1.0 - z_gate[j]) * agent.h[j] + z_gate[j] * h_hat[j];

                const out = [];
                for (let j = 0; j < 6; j++) {
                    let val = policy.b2[j];
                    for (let k = 0; k < 16; k++) val += agent.h[k] * policy.w2[k][j];
                    out.push(Math.tanh(val));
                }

                // Color-code by role (Phase 3 Visual Upgrade)
                // We use msg[0] magnitude to distinguish roles
                const activity = Math.abs(out[2]);
                if (activity > 0.6) agent.roleColor = '#fb7185'; // Pusher (High signal)
                else if (activity > 0.3) agent.roleColor = '#facc15'; // Scout (Medium signal)
                else agent.roleColor = '#22d3ee'; // Relay (Low signal)

                const fCurrent = { x: out[0] * 0.02, y: out[1] * 0.02 };
                const fDelayed = agent.motorQueue[1];
                agent.motorQueue[1] = agent.motorQueue[0]; agent.motorQueue[0] = fCurrent;
                agent.vx += (fDelayed.x / AGENT_MASS) * DT;
                agent.vy += (fDelayed.y / AGENT_MASS) * DT;
                for (let k = 0; k < 4; k++) agent.msg[k] = out[2+k];
            });

            // Physics (Grid Broadphase Parity)
            for (let i = 0; i < 100; i++) {
                const a = agentsRef.current[i];
                const gx = Math.max(0, Math.min(39, Math.floor(a.x / 20)));
                const gy = Math.max(0, Math.min(39, Math.floor(a.y / 20)));
                
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        const nx = gx + dx, ny = gy + dy;
                        if (nx >= 0 && nx < 40 && ny >= 0 && ny < 40) {
                            let curr = gridAgentHead[nx * 40 + ny];
                            while (curr !== -1) {
                                const j = curr;
                                if (i !== j) {
                                    const b = agentsRef.current[j];
                                    const dx_ = a.x - b.x, dy_ = a.y - b.y;
                                    const d = Math.sqrt(dx_*dx_ + dy_*dy_) + 1e-5;
                                    if (d < COLLISION_RADIUS) {
                                        const pushF = (COLLISION_RADIUS - d) * K_AGENT;
                                        a.vx += (dx_/d * pushF / AGENT_MASS) * DT;
                                        a.vy += (dy_/d * pushF / AGENT_MASS) * DT;
                                    }
                                }
                                curr = gridAgentNext[j];
                            }
                        }
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
                        const dx_ = a.x - obs.x, dy_ = a.y - obs.y;
                        const d = Math.sqrt(dx_*dx_ + dy_*dy_) + 1e-5;
                        const rSum = AGENT_RADIUS + 20.0;
                        if (d < rSum) {
                            const pushF = (rSum - d) * K_OBS;
                            a.vx += (dx_/d * pushF / AGENT_MASS) * DT; a.vy += (dy_/d * pushF / AGENT_MASS) * DT;
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
                    const dx_ = obj.x - obs.x, dy_ = obj.y - obs.y;
                    const d = Math.sqrt(dx_*dx_ + dy_*dy_) + 1e-5;
                    const rSum = OBJ_SIZE + 20.0;
                    if (d < rSum) {
                        const pushF = (rSum - d) * K_OBS;
                        obj.vx += (dx_/d * pushF / OBJ_MASS) * DT; obj.vy += (dy_/d * pushF / OBJ_MASS) * DT;
                    }
                }
            });
            obj.x += obj.vx * DT; obj.y += obj.vy * DT;
            if (obj.x < OBJ_SIZE) { obj.x = OBJ_SIZE; obj.vx *= -0.5; } if (obj.x > WIDTH - OBJ_SIZE) { obj.x = WIDTH - OBJ_SIZE; obj.vx *= -0.5; }
            if (obj.y < OBJ_SIZE) { obj.y = OBJ_SIZE; obj.vy *= -0.5; } if (obj.y > HEIGHT - OBJ_SIZE) { obj.y = HEIGHT - OBJ_SIZE; obj.vy *= -0.5; }
            if (Math.sqrt((obj.x - target.x)**2 + (obj.y - target.y)**2) < 40.0) { successRef.current = true; setSuccess(true); }
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
        
        // Agents (Color-coded by Role)
        agentsRef.current.forEach(a => {
            ctx.fillStyle = successRef.current ? '#94a3b8' : a.roleColor;
            ctx.beginPath(); ctx.arc(a.x, a.y, AGENT_RADIUS, 0, Math.PI*2); ctx.fill();
        });
        
        animationId = requestAnimationFrame(loop);
    };
    animationId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animationId);
  }, []);

  const setTarget = (nx: number, ny: number) => { 
      targetRef.current = { x: nx, y: ny }; 
      successRef.current = false; setSuccess(false); 
      trailRef.current = [];
      agentsRef.current.forEach(a => {
          a.h = new Array(16).fill(0); a.msg = new Array(4).fill(0);
          a.motorQueue = [{x:0, y:0}, {x:0, y:0}];
      });
      const obs = [];
      const dx = nx - 400, dy = ny - 250;
      const dist = Math.sqrt(dx*dx + dy*dy) + 1e-5;
      for (let i = 0; i < 3; i++) {
          const t = 0.3 + 0.4 * Math.random();
          const midX = 400 + dx * t, midY = 250 + dy * t;
          const px = -dy/dist, py = dx/dist;
          obs.push({ x: midX + px * (Math.random()-0.5) * 200, y: midY + py * (Math.random()-0.5) * 200, active: true });
      }
      obstaclesRef.current = obs;
  };

  const resetEnv = () => {
      objPosRef.current = { x: 400, y: 250, vx: 0, vy: 0 };
      successRef.current = false; setSuccess(false);
      targetRef.current = null; trailRef.current = []; obstaclesRef.current = [];
      agentsRef.current.forEach(a => {
          a.h = new Array(16).fill(0); a.msg = new Array(4).fill(0);
          a.motorQueue = [{x:0, y:0}, {x:0, y:0}];
      });
  };

  return { canvasRef, setTarget, resetEnv, agentCount: 100, brainActive, success };
};
