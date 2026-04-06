import React from 'react';
import { usePhysics } from './hooks/usePhysics';
import { Trash2, Target } from 'lucide-react';

const App: React.FC = () => {
  const { canvasRef, setTarget, resetEnv, brainActive, success } = usePhysics();

  const handleMouseDown = (e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    // Map mouse click to 800x500 internal canvas space
    const scaleX = 800 / rect.width;
    const scaleY = 500 / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    
    setTarget(x, y);
  };

  return (
    <div className="relative w-full h-screen bg-slate-900 text-slate-100 font-mono select-none overflow-hidden flex flex-col items-center justify-center">
      {/* HUD */}
      <div className="absolute top-6 left-6 z-10 p-4 bg-slate-800/80 backdrop-blur-md border border-slate-700 rounded-lg shadow-xl pointer-events-none w-64">
        <h1 className="text-xl font-bold text-cyan-400 tracking-tighter uppercase mb-4">Swarm AI v3.0</h1>
        <div className="space-y-2 text-xs">
          <div className="flex justify-between">
            <span className="opacity-50">BRAIN:</span>
            <span className={brainActive ? "text-cyan-400 font-bold" : "text-yellow-500 font-bold"}>
                {brainActive ? 'SYNCED NEURAL' : 'LOADING...'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="opacity-50">STATUS:</span>
            <span className={success ? "text-green-400 font-bold animate-pulse" : "text-cyan-400 font-bold"}>
                {success ? 'MISSION SUCCESS' : 'READY'}
            </span>
          </div>
        </div>
      </div>

      <div className="relative flex flex-col items-center gap-4">
        <canvas 
            ref={canvasRef} 
            width={800}
            height={500}
            onMouseDown={handleMouseDown}
            className="relative bg-slate-950 border border-slate-700 rounded-lg shadow-2xl cursor-crosshair active:cursor-grabbing overflow-hidden"
        />

        {/* Controls */}
        <div className="flex gap-4 p-2 bg-slate-800/90 backdrop-blur-md border border-slate-700 rounded-2xl shadow-2xl">
            <div className="p-3 bg-cyan-500/20 text-cyan-400 ring-1 ring-cyan-400 rounded-xl" title="Target Tool Active">
                <Target className="w-6 h-6" />
            </div>
            <div className="w-px h-8 bg-slate-700 my-auto mx-1" />
            <button onClick={resetEnv} className="p-3 hover:bg-red-500/20 text-red-400 rounded-xl transition-colors" title="Reset Environment">
                <Trash2 className="w-6 h-6" />
            </button>
        </div>
      </div>

      <div className="absolute bottom-6 text-[10px] text-slate-500 uppercase tracking-widest pointer-events-none flex gap-8">
        <span>Master Success-Aware Policy Active</span>
        {success ? <span className="text-green-400">Target Reached - Swarm Powering Down</span> : <span>Click anywhere to set mission target</span>}
      </div>
    </div>
  );
};

export default App;
