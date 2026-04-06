import { useEffect, useRef, useState } from 'react';
import policyData from '../assets/policy.json';

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

const K_AGENT = 0.02; // Agent-Agent spring stiffness
const K_OBJ = 0.05;   // Agent-Object spring stiffness

interface Vector { x: number; y: number; }

const normalize = (v: Vector): Vector => {
    const mag = Math.sqrt(v.x * v.x + v.y * v.y);
    return mag > 0.001 ? { x: v.x / mag, y: v.y / mag } : { x: 0, y: 0 };
};

export const usePhysics = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [brainActive, setBrainActive] = useState(false);
  const [success, setSuccess] = useState(false);
  
  // State
  const agentsRef = useRef<{x: number, y: number, vx: number, vy: number}[]>([]);
  const objPosRef = useRef<{x: number, y: number, vx: number, vy: number}>({ x: 400, y: 250, vx: 0, vy: 0 });
  const targetRef = useRef<Vector | null>(null);
  const policyRef = useRef<any>(null);

  useEffect(() => {
    // 1. Init
    const initialAgents = [];
    for (let i = 0; i < 100; i++) {
        const angle = Math.random() * 2.0 * Math.PI;
        const r = 60.0 + Math.random() * 40.0;
        initialAgents.push({
            x: 400 + Math.cos(angle) * r, y: 250 + Math.sin(angle) * r,
            vx: 0, vy: 0
        });
    }
    agentsRef.current = initialAgents;

    // 2. Load
    console.log("Marathon OpenAI-ES Policy Bundled & Loaded");
    policyRef.current = policyData;
    setBrainActive(true);

    // 3. Loop
    let animationId: number;
    const loop = () => {
        const ctx = canvasRef.current?.getContext('2d');
        if (!ctx) return;

        const obj = objPosRef.current;
        const target = targetRef.current;
        const policy = policyRef.current;

        if (target && policy && !success) {
            // Local Consensus
            let avgVX = 0, avgVY = 0;
            agentsRef.current.forEach(a => { avgVX += a.vx; avgVY += a.vy; });
            avgVX /= 100; avgVY /= 100;

            agentsRef.current.forEach((agent) => {
                // Sensors
                const toT = normalize({ x: target.x - obj.x, y: target.y - obj.y });
                const toO = normalize({ x: obj.x - agent.x, y: obj.y - agent.y });
                const rp = { x: (agent.x - obj.x) / 100.0, y: (agent.y - obj.y) / 100.0 };
                
                const wL = 1.0 / (1.0 + agent.x);
                const wR = 1.0 / (1.0 + (WIDTH - agent.x));
                const wT = 1.0 / (1.0 + agent.y);
                const wB = 1.0 / (1.0 + (HEIGHT - agent.y));

                const inpCorrect = [
                    toT.x, toT.y, toO.x, toO.y, rp.x, rp.y, agent.vx, agent.vy, avgVX, avgVY,
                    wL, wR, wT, wB, 
                    0.0, 0.0, 1.0, Math.sqrt(agent.vx*agent.vx + agent.vy*agent.vy)
                ];

                // Inference (18 -> 64 -> 2)
                const h1 = [];
                for (let j = 0; j < 64; j++) {
                    let val = policy.b1[j];
                    for (let k = 0; k < 18; k++) val += inpCorrect[k] * policy.W1[k][j];
                    h1.push(Math.tanh(val));
                }
                let outX = policy.b2[0], outY = policy.b2[1];
                for (let k = 0; k < 64; k++) {
                    outX += h1[k] * policy.W2[k][0];
                    outY += h1[k] * policy.W2[k][1];
                }
                outX = Math.tanh(outX); outY = Math.tanh(outY);

                // Physics (Force)
                agent.vx += (outX * 0.02 / AGENT_MASS) * DT;
            });

            // Physics (Hooke's Law Soft Contacts)
            for (let i = 0; i < 100; i++) {
                const a = agentsRef.current[i];
                for (let j = i + 1; j < 100; j++) {
                    const b = agentsRef.current[j];
                    const dx = a.x - b.x, dy = a.y - b.y;
                    const d = Math.sqrt(dx*dx + dy*dy) + 1e-5;
                    if (d < COLLISION_RADIUS) {
                        const overlap = COLLISION_RADIUS - d;
                        const pushF = overlap * K_AGENT;
                        const fx = (dx/d) * pushF;
                        const fy = (dy/d) * pushF;
                        a.vx += (fx / AGENT_MASS) * DT; a.vy += (fy / AGENT_MASS) * DT;
                        b.vx -= (fx / AGENT_MASS) * DT; b.vy -= (fy / AGENT_MASS) * DT;
                    }
                }
                
                // Agent-Object Soft Collision
                const odx = a.x - obj.x, ody = a.y - obj.y;
                if (Math.abs(odx) < OBJ_SIZE + AGENT_RADIUS && Math.abs(ody) < OBJ_SIZE + AGENT_RADIUS) {
                    const px = OBJ_SIZE + AGENT_RADIUS - Math.abs(odx);
                    const py = OBJ_SIZE + AGENT_RADIUS - Math.abs(ody);
                    let fx = 0, fy = 0;
                    if (px < py) {
                        fx = (odx >= 0 ? 1 : -1) * px * K_OBJ;
                    } else {
                        fy = (ody >= 0 ? 1 : -1) * py * K_OBJ;
                    }
                    a.vx += (fx / AGENT_MASS) * DT; a.vy += (fy / AGENT_MASS) * DT;
                    obj.vx -= (fx / OBJ_MASS) * DT; obj.vy -= (fy / OBJ_MASS) * DT;
                }

                // Agent Integration
                a.vx *= (1.0 - FRICTION_AIR);
                a.vy *= (1.0 - FRICTION_AIR);
                a.x += a.vx * DT;
                a.y += a.vy * DT;

                // Agent Boundary
                if (a.x < 0) { a.x = 0; a.vx *= -0.5; }
                if (a.x > WIDTH) { a.x = WIDTH; a.vx *= -0.5; }
                if (a.y < 0) { a.y = 0; a.vy *= -0.5; }
                if (a.y > HEIGHT) { a.y = HEIGHT; a.vy *= -0.5; }
            }

            // Object Integration
            obj.vx *= (1.0 - FRICTION_AIR);
            obj.vy *= (1.0 - FRICTION_AIR);
            obj.x += obj.vx * DT;
            obj.y += obj.vy * DT;

            // Object Boundary
            if (obj.x < OBJ_SIZE) { obj.x = OBJ_SIZE; obj.vx *= -0.5; }
            if (obj.x > WIDTH - OBJ_SIZE) { obj.x = WIDTH - OBJ_SIZE; obj.vx *= -0.5; }
            if (obj.y < OBJ_SIZE) { obj.y = OBJ_SIZE; obj.vy *= -0.5; }
            if (obj.y > HEIGHT - OBJ_SIZE) { obj.y = HEIGHT - OBJ_SIZE; obj.vy *= -0.5; }

            // Success Detection
            const distToTarget = Math.sqrt((obj.x - target.x)**2 + (obj.y - target.y)**2);
            if (distToTarget < 40.0) setSuccess(true);
        }

        // RENDER
        ctx.clearRect(0, 0, WIDTH, HEIGHT);
        // Target
        if (target) {
            ctx.fillStyle = success ? '#22c55e' : '#f97316';
            ctx.beginPath(); ctx.arc(target.x, target.y, 12, 0, Math.PI*2); ctx.fill();
        }
        // Object
        ctx.strokeStyle = success ? '#22c55e' : '#fb923c'; ctx.lineWidth = 3;
        ctx.strokeRect(obj.x - OBJ_SIZE, obj.y - OBJ_SIZE, OBJ_SIZE*2, OBJ_SIZE*2);
        // Agents
        ctx.fillStyle = success ? '#94a3b8' : '#22d3ee';
        agentsRef.current.forEach(a => {
            ctx.beginPath(); ctx.arc(a.x, a.y, AGENT_RADIUS, 0, Math.PI*2); ctx.fill();
        });

        animationId = requestAnimationFrame(loop);
    };

    animationId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animationId);
  }, [success]);

  const setTarget = (x: number, y: number) => { targetRef.current = { x, y }; setSuccess(false); };
  const resetEnv = () => {
      objPosRef.current = { x: 400, y: 250, vx: 0, vy: 0 };
      setSuccess(false);
      targetRef.current = null;
  };

  return { canvasRef, setTarget, resetEnv, agentCount: 100, brainActive, success };
};
