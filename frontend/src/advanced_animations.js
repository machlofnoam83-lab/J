/**
 * Advanced Animations - MARK 85 HUD
 * Grid, Particles, Reactor 3D, Memory, Scanlines
 */

class GridAnimation {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.resize();
        this.offset = 0;
        window.addEventListener('resize', () => this.resize());
        this.animate();
    }
    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }
    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.strokeStyle = 'rgba(0, 210, 255, 0.04)';
        this.ctx.lineWidth = 1;
        
        const gridSize = 40;
        this.offset = (this.offset + 0.5) % gridSize;
        
        // Vertical lines with perspective
        for (let x = -gridSize; x < this.canvas.width + gridSize; x += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(x + this.offset, 0);
            this.ctx.lineTo(x + this.offset * 0.5, this.canvas.height);
            this.ctx.stroke();
        }
        // Horizontal
        for (let y = 0; y < this.canvas.height; y += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }
        
        // Hex overlay faint
        this.ctx.strokeStyle = 'rgba(0, 210, 255, 0.02)';
        const hexSize = 80;
        for (let y = 0; y < this.canvas.height; y += hexSize * 0.866) {
            for (let x = 0; x < this.canvas.width; x += hexSize * 1.5) {
                const offsetX = (Math.floor(y / (hexSize * 0.866)) % 2) * hexSize * 0.75;
                this.drawHex(x + offsetX, y, hexSize * 0.4);
            }
        }
        
        requestAnimationFrame(() => this.animate());
    }
    drawHex(x, y, size) {
        this.ctx.beginPath();
        for (let i = 0; i < 6; i++) {
            const angle = (Math.PI / 3) * i;
            const hx = x + size * Math.cos(angle);
            const hy = y + size * Math.sin(angle);
            if (i === 0) this.ctx.moveTo(hx, hy);
            else this.ctx.lineTo(hx, hy);
        }
        this.ctx.closePath();
        this.ctx.stroke();
    }
}

class ParticlesAnimation {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.particles = [];
        this.resize();
        for (let i = 0; i < 80; i++) this.particles.push(this.createParticle());
        window.addEventListener('resize', () => this.resize());
        this.animate();
    }
    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }
    createParticle() {
        return {
            x: Math.random() * this.canvas.width,
            y: Math.random() * this.canvas.height,
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            size: Math.random() * 2 + 0.5,
            opacity: Math.random() * 0.5 + 0.1,
            color: Math.random() > 0.5 ? '0, 210, 255' : '0, 255, 170'
        };
    }
    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.particles.forEach((p, i) => {
            p.x += p.vx;
            p.y += p.vy;
            
            if (p.x < 0 || p.x > this.canvas.width) p.vx *= -1;
            if (p.y < 0 || p.y > this.canvas.height) p.vy *= -1;
            
            // Draw particle
            this.ctx.fillStyle = `rgba(${p.color}, ${p.opacity})`;
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Connections
            for (let j = i + 1; j < this.particles.length; j++) {
                const p2 = this.particles[j];
                const dist = Math.hypot(p.x - p2.x, p.y - p2.y);
                if (dist < 120) {
                    this.ctx.strokeStyle = `rgba(0, 210, 255, ${0.1 * (1 - dist / 120)})`;
                    this.ctx.lineWidth = 0.5;
                    this.ctx.beginPath();
                    this.ctx.moveTo(p.x, p.y);
                    this.ctx.lineTo(p2.x, p2.y);
                    this.ctx.stroke();
                }
            }
        });
        
        requestAnimationFrame(() => this.animate());
    }
}

class ScanlineAnimation {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.resize();
        this.scanY = 0;
        window.addEventListener('resize', () => this.resize());
        this.animate();
    }
    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }
    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Scanline
        this.ctx.fillStyle = 'rgba(0, 210, 255, 0.03)';
        this.ctx.fillRect(0, this.scanY, this.canvas.width, 2);
        
        // Glow
        const grad = this.ctx.createLinearGradient(0, this.scanY - 20, 0, this.scanY + 20);
        grad.addColorStop(0, 'transparent');
        grad.addColorStop(0.5, 'rgba(0, 210, 255, 0.05)');
        grad.addColorStop(1, 'transparent');
        this.ctx.fillStyle = grad;
        this.ctx.fillRect(0, this.scanY - 20, this.canvas.width, 40);
        
        this.scanY += 1.5;
        if (this.scanY > this.canvas.height) this.scanY = -20;
        
        requestAnimationFrame(() => this.animate());
    }
}

class ReactorAnimationAdvanced {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.width = canvas.width;
        this.height = canvas.height;
        this.centerX = this.width / 2;
        this.centerY = this.height / 2;
        this.time = 0;
        this.mode = 'idle';
        this.particles = [];
        for (let i = 0; i < 30; i++) {
            this.particles.push({
                angle: Math.random() * Math.PI * 2,
                radius: 30 + Math.random() * 80,
                speed: 0.002 + Math.random() * 0.005,
                size: Math.random() * 2 + 0.5,
                opacity: Math.random() * 0.8 + 0.2
            });
        }
        this.animate();
    }
    setMode(mode) {
        this.mode = mode;
        const speeds = { idle: 0.003, listening: 0.015, speaking: 0.01, thinking: 0.008 };
        const speed = speeds[mode] || 0.003;
        this.particles.forEach(p => p.speed = speed + Math.random() * 0.003);
    }
    draw() {
        this.ctx.clearRect(0, 0, this.width, this.height);
        
        // Glow
        const glowGrad = this.ctx.createRadialGradient(this.centerX, this.centerY, 0, this.centerX, this.centerY, 100);
        if (this.mode === 'listening') {
            glowGrad.addColorStop(0, 'rgba(255, 204, 0, 0.3)');
            glowGrad.addColorStop(0.5, 'rgba(255, 204, 0, 0.08)');
        } else if (this.mode === 'speaking') {
            glowGrad.addColorStop(0, 'rgba(0, 210, 255, 0.4)');
            glowGrad.addColorStop(0.5, 'rgba(0, 255, 170, 0.15)');
        } else {
            glowGrad.addColorStop(0, 'rgba(0, 210, 255, 0.2)');
            glowGrad.addColorStop(0.5, 'rgba(0, 210, 255, 0.05)');
        }
        glowGrad.addColorStop(1, 'transparent');
        this.ctx.fillStyle = glowGrad;
        this.ctx.fillRect(0, 0, this.width, this.height);
        
        // Outer tech ring
        this.ctx.strokeStyle = this.mode === 'listening' ? 'rgba(255,204,0,0.5)' : 'rgba(0,210,255,0.3)';
        this.ctx.lineWidth = 1.5;
        this.ctx.beginPath();
        this.ctx.arc(this.centerX, this.centerY, 95, 0, Math.PI * 2);
        this.ctx.stroke();
        
        // Dashed rings rotating
        this.ctx.save();
        this.ctx.translate(this.centerX, this.centerY);
        this.ctx.rotate(this.time * 0.008);
        this.ctx.strokeStyle = 'rgba(0,210,255,0.25)';
        this.ctx.setLineDash([3, 8]);
        this.ctx.beginPath();
        this.ctx.arc(0, 0, 75, 0, Math.PI * 2);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
        this.ctx.restore();
        
        this.ctx.save();
        this.ctx.translate(this.centerX, this.centerY);
        this.ctx.rotate(-this.time * 0.012);
        this.ctx.strokeStyle = 'rgba(168,85,247,0.2)';
        this.ctx.setLineDash([2, 6]);
        this.ctx.beginPath();
        this.ctx.arc(0, 0, 115, 0, Math.PI * 2);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
        this.ctx.restore();
        
        // Inner ring
        this.ctx.strokeStyle = 'rgba(0,255,170,0.4)';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.arc(this.centerX, this.centerY, 55, 0, Math.PI * 2);
        this.ctx.stroke();
        
        // Triangle core
        this.ctx.save();
        this.ctx.translate(this.centerX, this.centerY);
        this.ctx.rotate(this.time * 0.02);
        this.ctx.fillStyle = this.mode === 'listening' ? '#ffcc00' : '#00d2ff';
        this.ctx.shadowColor = this.ctx.fillStyle;
        this.ctx.shadowBlur = 20;
        this.ctx.beginPath();
        for (let i = 0; i < 3; i++) {
            const angle = (Math.PI * 2 / 3) * i - Math.PI / 2;
            const x = Math.cos(angle) * 22;
            const y = Math.sin(angle) * 22;
            if (i === 0) this.ctx.moveTo(x, y);
            else this.ctx.lineTo(x, y);
        }
        this.ctx.closePath();
        this.ctx.fill();
        this.ctx.shadowBlur = 0;
        this.ctx.restore();
        
        // Core circle pulsing
        const pulse = 1 + Math.sin(this.time * 0.06) * 0.12;
        const coreSize = this.mode === 'speaking' ? 14 * pulse : 11;
        this.ctx.fillStyle = this.mode === 'listening' ? 'rgba(255,204,0,0.9)' : 'rgba(0,210,255,0.9)';
        this.ctx.shadowColor = this.ctx.fillStyle;
        this.ctx.shadowBlur = 25;
        this.ctx.beginPath();
        this.ctx.arc(this.centerX, this.centerY, coreSize, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.shadowBlur = 0;
        
        // Particles orbit
        this.particles.forEach(p => {
            p.angle += p.speed;
            if (this.mode === 'speaking') {
                p.angle += Math.sin(this.time * 0.01 + p.angle) * 0.008;
            }
            const x = this.centerX + Math.cos(p.angle) * p.radius;
            const y = this.centerY + Math.sin(p.angle) * p.radius;
            this.ctx.fillStyle = `rgba(${this.mode === 'listening' ? '255, 204, 0' : '0, 210, 255'}, ${p.opacity})`;
            this.ctx.beginPath();
            this.ctx.arc(x, y, p.size, 0, Math.PI * 2);
            this.ctx.fill();
        });
        
        // Energy spokes when speaking
        if (this.mode === 'speaking' || this.mode === 'thinking') {
            this.ctx.strokeStyle = this.mode === 'thinking' ? 'rgba(168,85,247,0.5)' : 'rgba(0,255,170,0.5)';
            this.ctx.lineWidth = 1;
            for (let i = 0; i < 12; i++) {
                const angle = (Math.PI * 2 / 12) * i + this.time * 0.02;
                const inner = 28;
                const outer = 45 + Math.sin(this.time * 0.08 + i) * 6;
                this.ctx.beginPath();
                this.ctx.moveTo(this.centerX + Math.cos(angle) * inner, this.centerY + Math.sin(angle) * inner);
                this.ctx.lineTo(this.centerX + Math.cos(angle) * outer, this.centerY + Math.sin(angle) * outer);
                this.ctx.stroke();
            }
        }
    }
    animate() {
        this.time++;
        this.draw();
        requestAnimationFrame(() => this.animate());
    }
}

class MemoryOrbAnimation {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.width = canvas.width;
        this.height = canvas.height;
        this.centerX = this.width / 2;
        this.centerY = this.height / 2;
        this.time = 0;
        this.nodes = [];
        for (let i = 0; i < 12; i++) {
            const angle = (Math.PI * 2 / 12) * i;
            this.nodes.push({
                angle: angle,
                radius: 30 + Math.random() * 10,
                baseRadius: 30,
                speed: 0.01 + Math.random() * 0.01,
                size: 3
            });
        }
        this.animate();
    }
    animate() {
        this.time++;
        this.ctx.clearRect(0, 0, this.width, this.height);
        
        // Center glow
        const grad = this.ctx.createRadialGradient(this.centerX, this.centerY, 0, this.centerX, this.centerY, 40);
        grad.addColorStop(0, 'rgba(0,210,255,0.3)');
        grad.addColorStop(1, 'transparent');
        this.ctx.fillStyle = grad;
        this.ctx.fillRect(0, 0, this.width, this.height);
        
        // Nodes orbiting
        this.nodes.forEach(n => {
            n.angle += n.speed;
            n.radius = n.baseRadius + Math.sin(this.time * 0.05 + n.angle) * 5;
            const x = this.centerX + Math.cos(n.angle) * n.radius;
            const y = this.centerY + Math.sin(n.angle) * n.radius;
            
            // Line to center
            this.ctx.strokeStyle = 'rgba(0,210,255,0.15)';
            this.ctx.lineWidth = 0.5;
            this.ctx.beginPath();
            this.ctx.moveTo(this.centerX, this.centerY);
            this.ctx.lineTo(x, y);
            this.ctx.stroke();
            
            // Node
            this.ctx.fillStyle = 'rgba(0,210,255,0.8)';
            this.ctx.beginPath();
            this.ctx.arc(x, y, n.size, 0, Math.PI * 2);
            this.ctx.fill();
        });
        
        // Center core
        this.ctx.fillStyle = 'rgba(0,210,255,0.9)';
        this.ctx.shadowColor = '#00d2ff';
        this.ctx.shadowBlur = 15;
        this.ctx.beginPath();
        this.ctx.arc(this.centerX, this.centerY, 12, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.shadowBlur = 0;
        
        requestAnimationFrame(() => this.animate());
    }
}

// Init all
document.addEventListener('DOMContentLoaded', () => {
    const gridCanvas = document.getElementById('gridCanvas');
    const particlesCanvas = document.getElementById('particlesCanvas');
    const scanlineCanvas = document.getElementById('scanlineCanvas');
    const reactorCanvas = document.getElementById('reactorCanvas');
    const memoryCanvas = document.getElementById('memoryCanvas');
    
    if (gridCanvas) new GridAnimation(gridCanvas);
    if (particlesCanvas) new ParticlesAnimation(particlesCanvas);
    if (scanlineCanvas) new ScanlineAnimation(scanlineCanvas);
    if (reactorCanvas) {
        window.reactorAnim = new ReactorAnimationAdvanced(reactorCanvas);
    }
    if (memoryCanvas) {
        new MemoryOrbAnimation(memoryCanvas);
    }
    
    // Orb canvas for orb mode
    const orbCanvas = document.getElementById('orbCanvas');
    if (orbCanvas) {
        new ReactorAnimationAdvanced(orbCanvas);
    }
    
    console.log('[Advanced Animations] All initialized - MARK 85');
});
