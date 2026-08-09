/**
 * Adiel Junior - Reactor Animations
 * Iron Man arc reactor canvas animation
 */

class ReactorAnimation {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.width = canvas.width;
        this.height = canvas.height;
        this.centerX = this.width / 2;
        this.centerY = this.height / 2;
        this.time = 0;
        this.mode = 'idle'; // idle, listening, speaking, thinking
        this.particles = [];
        this.initParticles();
        this.animate();
    }

    initParticles() {
        this.particles = [];
        for (let i = 0; i < 20; i++) {
            this.particles.push({
                angle: (Math.PI * 2 / 20) * i,
                radius: 35 + Math.random() * 20,
                speed: 0.002 + Math.random() * 0.003,
                size: 1 + Math.random() * 2,
                opacity: 0.3 + Math.random() * 0.7
            });
        }
    }

    setMode(mode) {
        this.mode = mode;
        if (mode === 'listening') {
            this.particles.forEach(p => p.speed = 0.01 + Math.random() * 0.01);
        } else if (mode === 'speaking') {
            this.particles.forEach(p => p.speed = 0.008 + Math.random() * 0.008);
        } else {
            this.particles.forEach(p => p.speed = 0.002 + Math.random() * 0.003);
        }
    }

    draw() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.width, this.height);

        // Background glow
        const glowGrad = ctx.createRadialGradient(this.centerX, this.centerY, 0, this.centerX, this.centerY, 60);
        if (this.mode === 'listening') {
            glowGrad.addColorStop(0, 'rgba(255, 204, 0, 0.3)');
            glowGrad.addColorStop(0.5, 'rgba(255, 204, 0, 0.1)');
            glowGrad.addColorStop(1, 'transparent');
        } else if (this.mode === 'speaking') {
            glowGrad.addColorStop(0, 'rgba(0, 210, 255, 0.4)');
            glowGrad.addColorStop(0.5, 'rgba(0, 210, 255, 0.15)');
            glowGrad.addColorStop(1, 'transparent');
        } else {
            glowGrad.addColorStop(0, 'rgba(0, 210, 255, 0.25)');
            glowGrad.addColorStop(0.5, 'rgba(0, 210, 255, 0.08)');
            glowGrad.addColorStop(1, 'transparent');
        }
        ctx.fillStyle = glowGrad;
        ctx.fillRect(0, 0, this.width, this.height);

        // Outer ring
        ctx.strokeStyle = this.mode === 'listening' ? 'rgba(255,204,0,0.6)' : 'rgba(0,210,255,0.4)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(this.centerX, this.centerY, 62, 0, Math.PI * 2);
        ctx.stroke();

        // Dashed middle ring - rotating
        ctx.save();
        ctx.translate(this.centerX, this.centerY);
        ctx.rotate(this.time * 0.01);
        ctx.strokeStyle = 'rgba(0,210,255,0.3)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 8]);
        ctx.beginPath();
        ctx.arc(0, 0, 52, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();

        // Inner ring
        ctx.strokeStyle = 'rgba(0,255,170,0.5)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(this.centerX, this.centerY, 38, 0, Math.PI * 2);
        ctx.stroke();

        // Core triangle - Iron Man style
        ctx.save();
        ctx.translate(this.centerX, this.centerY);
        ctx.rotate(this.time * 0.02);
        ctx.fillStyle = this.mode === 'listening' ? '#ffcc00' : '#00d2ff';
        ctx.shadowColor = this.mode === 'listening' ? '#ffcc00' : '#00d2ff';
        ctx.shadowBlur = 15;
        
        ctx.beginPath();
        for (let i = 0; i < 3; i++) {
            const angle = (Math.PI * 2 / 3) * i - Math.PI / 2;
            const x = Math.cos(angle) * 18;
            const y = Math.sin(angle) * 18;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.restore();

        // Core circle
        const corePulse = 1 + Math.sin(this.time * 0.05) * 0.1;
        if (this.mode === 'speaking') {
            corePulse * 1.3;
        }
        
        ctx.fillStyle = this.mode === 'listening' ? 'rgba(255,204,0,0.9)' : 'rgba(0,210,255,0.9)';
        ctx.shadowColor = ctx.fillStyle;
        ctx.shadowBlur = 20;
        ctx.beginPath();
        ctx.arc(this.centerX, this.centerY, 12 * (this.mode === 'speaking' ? corePulse : 1), 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;

        // Particles
        this.particles.forEach(p => {
            p.angle += p.speed;
            if (this.mode === 'speaking') {
                p.angle += Math.sin(this.time * 0.01 + p.angle) * 0.01;
            }
            
            const x = this.centerX + Math.cos(p.angle) * p.radius;
            const y = this.centerY + Math.sin(p.angle) * p.radius;
            
            ctx.fillStyle = `rgba(0, 210, 255, ${p.opacity})`;
            if (this.mode === 'listening') {
                ctx.fillStyle = `rgba(255, 204, 0, ${p.opacity})`;
            }
            ctx.beginPath();
            ctx.arc(x, y, p.size, 0, Math.PI * 2);
            ctx.fill();
        });

        // Energy lines - when speaking
        if (this.mode === 'speaking') {
            ctx.strokeStyle = 'rgba(0,255,170,0.6)';
            ctx.lineWidth = 1;
            for (let i = 0; i < 8; i++) {
                const angle = (Math.PI * 2 / 8) * i + this.time * 0.02;
                const inner = 20;
                const outer = 35 + Math.sin(this.time * 0.05 + i) * 5;
                ctx.beginPath();
                ctx.moveTo(
                    this.centerX + Math.cos(angle) * inner,
                    this.centerY + Math.sin(angle) * inner
                );
                ctx.lineTo(
                    this.centerX + Math.cos(angle) * outer,
                    this.centerY + Math.sin(angle) * outer
                );
                ctx.stroke();
            }
        }
    }

    animate() {
        this.time++;
        this.draw();
        requestAnimationFrame(() => this.animate());
    }
}

// Global instance
let reactorAnim = null;

document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('reactorCanvas');
    if (canvas) {
        reactorAnim = new ReactorAnimation(canvas);
        window.reactorAnim = reactorAnim; // expose for HUD control
    }
});
