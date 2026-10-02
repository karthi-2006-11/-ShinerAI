/**
 * ShinerAI: Photorealistic Interactive Aquarium Background Engine
 *
 * Implements a realistic freshwater aquarium simulation:
 * - 35-50 realistic goldfish with natural swimming undulation and multi-joint spine articulation
 * - Multiple loosely schooling fish groups with cohesion, separation, alignment, and wander
 * - Dynamic group mouse escape physics: nearby fish/groups notice cursor, accelerate smoothly,
 *   steer away with organic dispersion, and recover to calm slow swimming
 * - Swaying aquatic vegetation (Vallisneria ribbon grass and Amazon sword broad-leafs)
 * - Continuous rising bubble aerator streams with realistic buoyancy and horizontal drift
 * - Underwater volumetric light rays and water surface caustics
 * - Layered depth (background, midground, foreground, substrate rocks)
 * - Automatic performance throttling (Page Visibility API, requestAnimationFrame)
 * - Full accessibility compliance (prefers-reduced-motion, aria-hidden, pointer-events: none)
 */

(function () {
    "use strict";

    // -------------------------------------------------------------------------
    // Configuration & Simulation Constants
    // -------------------------------------------------------------------------
    const CONFIG = {
        // Fish population by device viewport
        fishCountDesktop: 42,
        fishCountTablet: 26,
        fishCountMobile: 16,
        groupCount: 5,

        // Calibrated swimming dynamics (pixels per frame at 60fps)
        calmSpeedMin: 0.7,
        calmSpeedMax: 1.3,
        escapeSpeedMax: 5.5,
        turnRateCalm: 0.05,
        turnRateEscape: 0.14,
        recoveryRate: 0.035, // Rate at which fish recover from escape to calm

        // Mouse interaction thresholds
        mouseNoticeRadius: 260, // Distance where fish notice approaching cursor
        mouseFleeRadius: 180,   // Distance where active acceleration begins
        mousePanicRadius: 80,   // Rapid burst escape threshold
        groupNoticeRadius: 300, // Distance to group center triggering group awareness

        // Aerator bubbles
        bubbleCountDesktop: 38,
        bubbleCountMobile: 18,
        bubbleEmitters: [0.08, 0.32, 0.68, 0.92], // Screen X proportions

        // Caustics & Sunbeams
        sunbeamCount: 4,
    };

    // Color palettes for realistic goldfish variations
    const FISH_PALETTES = [
        {
            // Classic Deep Goldfish
            dorsal: "#c2410c",
            flank: "#ea580c",
            belly: "#ffedd5",
            finBase: "rgba(234, 88, 12, 0.75)",
            finTip: "rgba(254, 215, 170, 0.35)",
            eyeIris: "#d97706",
        },
        {
            // Radiant Golden Amber
            dorsal: "#b45309",
            flank: "#f59e0b",
            belly: "#fef3c7",
            finBase: "rgba(245, 158, 11, 0.75)",
            finTip: "rgba(253, 230, 138, 0.35)",
            eyeIris: "#b45309",
        },
        {
            // Rich Orange Vermilion
            dorsal: "#991b1b",
            flank: "#f97316",
            belly: "#fff7ed",
            finBase: "rgba(249, 115, 22, 0.8)",
            finTip: "rgba(255, 237, 213, 0.4)",
            eyeIris: "#ea580c",
        },
        {
            // Pearl-Scale White & Gold
            dorsal: "#ea580c",
            flank: "#fed7aa",
            belly: "#ffffff",
            finBase: "rgba(251, 146, 60, 0.65)",
            finTip: "rgba(255, 255, 255, 0.3)",
            eyeIris: "#92400e",
        },
    ];

    // -------------------------------------------------------------------------
    // Utility Mathematical Helpers
    // -------------------------------------------------------------------------
    function randomRange(min, max) {
        return min + Math.random() * (max - min);
    }

    function clamp(val, min, max) {
        return Math.max(min, Math.min(max, val));
    }

    function lerp(a, b, t) {
        return a + (b - a) * t;
    }

    function lerpAngle(from, to, t) {
        let diff = (to - from) % (Math.PI * 2);
        if (diff < -Math.PI) diff += Math.PI * 2;
        if (diff > Math.PI) diff -= Math.PI * 2;
        return from + diff * t;
    }

    function distance(x1, y1, x2, y2) {
        const dx = x2 - x1;
        const dy = y2 - y1;
        return Math.sqrt(dx * dx + dy * dy);
    }

    // -------------------------------------------------------------------------
    // Simulation State
    // -------------------------------------------------------------------------
    let canvas = null;
    let ctx = null;
    let width = 0;
    let height = 0;
    let dpr = 1;
    let animId = null;
    let lastTime = 0;
    let simTime = 0;
    let isReducedMotion = false;
    let isTabVisible = true;

    // Mouse tracking
    const mouse = {
        x: -9999,
        y: -9999,
        targetX: -9999,
        targetY: -9999,
        vx: 0,
        vy: 0,
        active: false,
        lastMoved: 0,
    };

    // Environment entities
    let groups = [];
    let fishList = [];
    let bubbles = [];
    let plants = [];
    let rocks = [];
    let waterRipples = [];

    // -------------------------------------------------------------------------
    // Entity Constructors
    // -------------------------------------------------------------------------

    /**
     * School/Group manager holding local cluster centroids
     */
    class FishGroup {
        constructor(id, x, y) {
            this.id = id;
            this.x = x;
            this.y = y;
            this.vx = randomRange(-0.8, 0.8);
            this.vy = randomRange(-0.3, 0.3);
            this.targetX = x;
            this.targetY = y;
            this.wanderTimer = randomRange(0, 5);
            this.alertLevel = 0; // 0 (calm) to 1 (high alert)
            this.depth = randomRange(0.4, 0.95);
        }

        update(dt) {
            this.wanderTimer -= dt;
            if (this.wanderTimer <= 0) {
                this.wanderTimer = randomRange(4, 9);
                // Wander primarily in upper/lower/side areas rather than dead center
                const targetX = randomRange(width * 0.08, width * 0.92);
                const targetY = randomRange(height * 0.12, height * 0.88);
                this.targetX = targetX;
                this.targetY = targetY;
            }

            // Smoothly steer towards wander target
            const dx = this.targetX - this.x;
            const dy = this.targetY - this.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist > 20) {
                this.vx = lerp(this.vx, (dx / dist) * 0.9, 0.02);
                this.vy = lerp(this.vy, (dy / dist) * 0.6, 0.02);
            }

            // React to mouse proximity to group centroid
            if (mouse.active && !isReducedMotion) {
                const distToMouse = distance(this.x, this.y, mouse.x, mouse.y);
                if (distToMouse < CONFIG.groupNoticeRadius) {
                    const fleeIntensity = 1 - distToMouse / CONFIG.groupNoticeRadius;
                    this.alertLevel = Math.max(this.alertLevel, fleeIntensity);

                    // Push group center away from mouse
                    const escapeAngle = Math.atan2(this.y - mouse.y, this.x - mouse.x);
                    this.vx += Math.cos(escapeAngle) * fleeIntensity * 1.8;
                    this.vy += Math.sin(escapeAngle) * fleeIntensity * 1.2;
                }
            }

            // Decay alert level gradually
            this.alertLevel = Math.max(0, this.alertLevel - dt * 0.35);

            // Boundary repulsion
            const marginX = width * 0.06;
            const marginY = height * 0.08;
            if (this.x < marginX) this.vx += 0.08;
            if (this.x > width - marginX) this.vx -= 0.08;
            if (this.y < marginY) this.vy += 0.08;
            if (this.y > height - marginY) this.vy -= 0.08;

            this.x += this.vx;
            this.y += this.vy;
        }
    }

    /**
     * Individual Goldfish with physical body undulation & steering
     */
    class Goldfish {
        constructor(id, groupIndex, x, y) {
            this.id = id;
            this.groupIndex = groupIndex;
            this.x = x + randomRange(-60, 60);
            this.y = y + randomRange(-40, 40);

            // Depth determines scale, optical haze, and rendering layer
            this.z = randomRange(0.35, 1.0);
            this.scale = lerp(0.55, 1.05, (this.z - 0.35) / 0.65);
            this.baseLength = 48 * this.scale;
            this.baseWidth = 18 * this.scale;

            // Palette
            this.palette = FISH_PALETTES[Math.floor(Math.random() * FISH_PALETTES.length)];

            // Motion variables
            this.heading = randomRange(0, Math.PI * 2);
            this.targetHeading = this.heading;
            this.baseSpeed = randomRange(CONFIG.calmSpeedMin, CONFIG.calmSpeedMax) * (0.8 + this.z * 0.3);
            this.speed = this.baseSpeed;
            this.targetSpeed = this.baseSpeed;

            // Tail & spine oscillation
            this.tailPhase = randomRange(0, Math.PI * 2);
            this.tailFrequency = 3.5;
            this.fleeIntensity = 0;
            this.fleeTimer = 0;

            // Personal offsets to avoid uniform swimming
            this.personalOffset = randomRange(-0.4, 0.4);
            this.personalWanderTimer = randomRange(1, 4);
        }

        update(dt) {
            const group = groups[this.groupIndex];

            // 1. Schooling forces (Cohesion towards group center)
            const dxGroup = group.x - this.x;
            const dyGroup = group.y - this.y;
            const distGroup = Math.sqrt(dxGroup * dxGroup + dyGroup * dyGroup);

            let steerX = 0;
            let steerY = 0;

            if (distGroup > 40) {
                steerX += (dxGroup / distGroup) * 0.45;
                steerY += (dyGroup / distGroup) * 0.35;
            }

            // 2. Separation force (avoid crowding nearby fish in the same layer)
            for (let i = 0; i < fishList.length; i++) {
                const other = fishList[i];
                if (other === this) continue;
                if (Math.abs(other.z - this.z) > 0.3) continue; // Only separate within similar depth

                const sepDx = this.x - other.x;
                const sepDy = this.y - other.y;
                const sepDist = Math.sqrt(sepDx * sepDx + sepDy * sepDy);
                const minComfortDist = 32 * this.scale;

                if (sepDist > 0 && sepDist < minComfortDist) {
                    const repulse = (minComfortDist - sepDist) / minComfortDist;
                    steerX += (sepDx / sepDist) * repulse * 0.8;
                    steerY += (sepDy / sepDist) * repulse * 0.8;
                }
            }

            // 3. Alignment with group velocity
            steerX += group.vx * 0.3;
            steerY += group.vy * 0.3;

            // 4. Boundary repulsion
            const padX = width * 0.05;
            const padY = height * 0.07;
            if (this.x < padX) steerX += ((padX - this.x) / padX) * 1.2;
            if (this.x > width - padX) steerX -= ((this.x - (width - padX)) / padX) * 1.2;
            if (this.y < padY) steerY += ((padY - this.y) / padY) * 1.2;
            if (this.y > height - padY) steerY -= ((this.y - (height - padY)) / padY) * 1.2;

            // 5. Individual wander noise
            this.personalWanderTimer -= dt;
            if (this.personalWanderTimer <= 0) {
                this.personalWanderTimer = randomRange(2, 5);
                this.personalOffset = randomRange(-0.35, 0.35);
            }
            steerX += Math.cos(this.heading + Math.PI / 2) * this.personalOffset * 0.25;
            steerY += Math.sin(this.heading + Math.PI / 2) * this.personalOffset * 0.25;

            // 6. Interactive Mouse Escape Physics
            this.targetSpeed = this.baseSpeed;
            let currentTurnRate = CONFIG.turnRateCalm;

            if (mouse.active && !isReducedMotion) {
                const distToMouse = distance(this.x, this.y, mouse.x, mouse.y);

                if (distToMouse < CONFIG.mouseNoticeRadius) {
                    const escapeAngle = Math.atan2(this.y - mouse.y, this.x - mouse.x);

                    // Add lateral scatter so escaping fish don't all flee on identical vectors
                    const personalScatter = Math.sin(this.id * 1.7) * 0.32;
                    const finalEscapeAngle = escapeAngle + personalScatter;

                    // Strength increases non-linearly as cursor approaches
                    let force = 0;
                    if (distToMouse < CONFIG.mousePanicRadius) {
                        // Extreme burst escape
                        force = 1.0;
                        this.targetSpeed = CONFIG.escapeSpeedMax * (0.85 + this.z * 0.3);
                        currentTurnRate = CONFIG.turnRateEscape * 1.2;
                    } else if (distToMouse < CONFIG.mouseFleeRadius) {
                        // Clear acceleration away
                        force = 1 - (distToMouse - CONFIG.mousePanicRadius) / (CONFIG.mouseFleeRadius - CONFIG.mousePanicRadius);
                        this.targetSpeed = lerp(this.baseSpeed * 2.2, CONFIG.escapeSpeedMax, force);
                        currentTurnRate = CONFIG.turnRateEscape;
                    } else {
                        // Early notice and gentle avoidance
                        force = (1 - (distToMouse - CONFIG.mouseFleeRadius) / (CONFIG.mouseNoticeRadius - CONFIG.mouseFleeRadius)) * 0.4;
                        this.targetSpeed = this.baseSpeed * (1 + force * 1.2);
                        currentTurnRate = CONFIG.turnRateCalm * 1.5;
                    }

                    steerX += Math.cos(finalEscapeAngle) * force * 3.5;
                    steerY += Math.sin(finalEscapeAngle) * force * 2.5;

                    this.fleeIntensity = Math.max(this.fleeIntensity, force);
                    this.fleeTimer = 1.8; // Maintain alert for brief window
                }
            }

            // Group alert contagion: if group is startled, members stay somewhat faster
            if (group.alertLevel > 0.1 && this.targetSpeed <= this.baseSpeed) {
                this.targetSpeed = this.baseSpeed * (1 + group.alertLevel * 0.8);
            }

            // Smoothly steer heading toward combined force vector
            if (Math.abs(steerX) > 0.001 || Math.abs(steerY) > 0.001) {
                this.targetHeading = Math.atan2(steerY, steerX);
            }

            this.heading = lerpAngle(this.heading, this.targetHeading, currentTurnRate);

            // Speed recovery easing
            if (this.speed < this.targetSpeed) {
                this.speed = lerp(this.speed, this.targetSpeed, 0.12); // Rapid acceleration
            } else {
                this.speed = lerp(this.speed, this.targetSpeed, CONFIG.recoveryRate); // Gentle deceleration
            }

            if (isReducedMotion) {
                this.speed = this.baseSpeed * 0.18; // Very slow serene drift
            }

            // Tail frequency scales with swimming velocity
            this.tailFrequency = 2.0 + this.speed * 2.2;
            this.tailPhase += dt * this.tailFrequency * (isReducedMotion ? 0.2 : 1.0);

            // Integrate position
            this.x += Math.cos(this.heading) * this.speed;
            this.y += Math.sin(this.heading) * this.speed;

            // Decay flee intensity
            this.fleeTimer = Math.max(0, this.fleeTimer - dt);
            if (this.fleeTimer <= 0) {
                this.fleeIntensity = Math.max(0, this.fleeIntensity - dt * 0.8);
            }
        }

        draw(ctx) {
            ctx.save();
            ctx.translate(this.x, this.y);
            ctx.rotate(this.heading);

            const L = this.baseLength;
            const W = this.baseWidth;
            const wave = Math.sin(this.tailPhase);
            const waveTail = Math.sin(this.tailPhase - 0.7);

            // Depth atmospheric hazing (distant fish are slightly desaturated and softer)
            ctx.globalAlpha = lerp(0.68, 0.98, this.z);

            // -------------------------------------------------------------
            // 1. Caudal Fin (Tail) - Translucent, Flowing Double-Lobed Tail
            // -------------------------------------------------------------
            ctx.save();
            ctx.translate(-L * 0.42, 0);

            const tailSpread = W * 1.35;
            const tailSwing = waveTail * (W * 0.45 + this.fleeIntensity * W * 0.2);

            const tailGrad = ctx.createLinearGradient(0, 0, -L * 0.65, tailSwing);
            tailGrad.addColorStop(0, this.palette.finBase);
            tailGrad.addColorStop(0.65, this.palette.flank);
            tailGrad.addColorStop(1, this.palette.finTip);

            ctx.fillStyle = tailGrad;
            ctx.beginPath();
            ctx.moveTo(0, 0);
            // Upper lobe
            ctx.bezierCurveTo(-L * 0.25, -tailSpread * 0.5, -L * 0.5, -tailSpread + tailSwing * 0.4, -L * 0.65, -tailSpread * 0.75 + tailSwing);
            // Middle cleft
            ctx.quadraticCurveTo(-L * 0.45, tailSwing * 0.6, -L * 0.68, tailSwing);
            // Lower lobe
            ctx.bezierCurveTo(-L * 0.5, tailSpread * 0.75 + tailSwing * 0.4, -L * 0.25, tailSpread * 0.5, 0, 0);
            ctx.fill();

            // Delicate fin rays
            ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
            ctx.lineWidth = 0.8;
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(-L * 0.62, -tailSpread * 0.6 + tailSwing);
            ctx.moveTo(0, 0);
            ctx.lineTo(-L * 0.65, tailSwing);
            ctx.moveTo(0, 0);
            ctx.lineTo(-L * 0.62, tailSpread * 0.6 + tailSwing);
            ctx.stroke();

            ctx.restore();

            // -------------------------------------------------------------
            // 2. Dorsal Fin (Top crest)
            // -------------------------------------------------------------
            ctx.save();
            ctx.translate(-L * 0.05, -W * 0.42);
            ctx.fillStyle = this.palette.finBase;
            ctx.beginPath();
            ctx.moveTo(L * 0.18, 0);
            ctx.quadraticCurveTo(0, -W * 0.55, -L * 0.22, -W * 0.2);
            ctx.quadraticCurveTo(-L * 0.1, 0, 0, 0);
            ctx.fill();
            ctx.restore();

            // -------------------------------------------------------------
            // 3. Ventral / Pelvic Fin (Bottom)
            // -------------------------------------------------------------
            ctx.save();
            ctx.translate(-L * 0.12, W * 0.38);
            ctx.fillStyle = this.palette.finTip;
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.quadraticCurveTo(-L * 0.12, W * 0.35, -L * 0.2, W * 0.2);
            ctx.quadraticCurveTo(-L * 0.08, 0, 0, 0);
            ctx.fill();
            ctx.restore();

            // -------------------------------------------------------------
            // 4. Fish Body (Spindle contour with subtle undulating spine)
            // -------------------------------------------------------------
            const midWave = wave * W * 0.14;
            const peduncleWave = waveTail * W * 0.28;

            const bodyGrad = ctx.createLinearGradient(0, -W * 0.6, 0, W * 0.6);
            bodyGrad.addColorStop(0, this.palette.dorsal);
            bodyGrad.addColorStop(0.35, this.palette.flank);
            bodyGrad.addColorStop(0.85, this.palette.belly);
            bodyGrad.addColorStop(1, "rgba(255, 255, 255, 0.85)");

            ctx.fillStyle = bodyGrad;
            ctx.beginPath();
            // Snout
            ctx.moveTo(L * 0.48, 0);
            // Upper dorsal curve
            ctx.bezierCurveTo(L * 0.3, -W * 0.55, -L * 0.05, -W * 0.65 + midWave, -L * 0.3, -W * 0.3 + peduncleWave);
            // Peduncle tip (tail connection)
            ctx.lineTo(-L * 0.45, peduncleWave);
            // Lower ventral curve (belly)
            ctx.bezierCurveTo(-L * 0.3, W * 0.35 + peduncleWave, -L * 0.05, W * 0.62 + midWave, L * 0.3, W * 0.45);
            ctx.closePath();
            ctx.fill();

            // Pearlescent scale sheen highlight along upper flank
            ctx.save();
            ctx.strokeStyle = "rgba(255, 255, 255, 0.45)";
            ctx.lineWidth = 1.2 * this.scale;
            ctx.beginPath();
            ctx.moveTo(L * 0.35, -W * 0.15);
            ctx.quadraticCurveTo(0, -W * 0.35 + midWave, -L * 0.25, -W * 0.12 + peduncleWave);
            ctx.stroke();
            ctx.restore();

            // -------------------------------------------------------------
            // 5. Pectoral Fin (Side flapping fin)
            // -------------------------------------------------------------
            ctx.save();
            ctx.translate(L * 0.14, W * 0.18);
            const finFlap = Math.sin(this.tailPhase * 1.5) * 0.28;
            ctx.rotate(0.35 + finFlap);
            ctx.fillStyle = this.palette.finTip;
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.quadraticCurveTo(L * 0.08, W * 0.32, -L * 0.06, W * 0.42);
            ctx.quadraticCurveTo(-L * 0.08, W * 0.18, 0, 0);
            ctx.fill();
            ctx.restore();

            // -------------------------------------------------------------
            // 6. Anatomical Eye & Gill Plate
            // -------------------------------------------------------------
            // Subtle gill slit
            ctx.strokeStyle = "rgba(180, 83, 9, 0.35)";
            ctx.lineWidth = 1.0 * this.scale;
            ctx.beginPath();
            ctx.arc(L * 0.22, 0, W * 0.38, -Math.PI * 0.45, Math.PI * 0.35);
            ctx.stroke();

            // Eye (Iris + Pupil + White Catchlight)
            const eyeX = L * 0.34;
            const eyeY = -W * 0.16;
            const eyeR = 2.4 * this.scale;

            // Amber Iris
            ctx.fillStyle = this.palette.eyeIris;
            ctx.beginPath();
            ctx.arc(eyeX, eyeY, eyeR, 0, Math.PI * 2);
            ctx.fill();

            // Dark Pupil
            ctx.fillStyle = "#0f172a";
            ctx.beginPath();
            ctx.arc(eyeX + 0.3, eyeY, eyeR * 0.65, 0, Math.PI * 2);
            ctx.fill();

            // Specular catchlight reflection
            ctx.fillStyle = "#ffffff";
            ctx.beginPath();
            ctx.arc(eyeX - 0.6, eyeY - 0.6, eyeR * 0.3, 0, Math.PI * 2);
            ctx.fill();

            ctx.restore();
        }
    }

    /**
     * Aquatic Plant (Vallisneria ribbon or Amazon Sword broad-leaf)
     */
    class AquaticPlant {
        constructor(x, y, type, heightRatio) {
            this.x = x;
            this.y = y;
            this.type = type; // "ribbon" or "broad"
            this.height = heightRatio * height * randomRange(0.65, 1.05);
            this.bladeCount = type === "ribbon" ? Math.floor(randomRange(5, 9)) : Math.floor(randomRange(4, 7));
            this.blades = [];
            this.phase = randomRange(0, Math.PI * 2);

            for (let i = 0; i < this.bladeCount; i++) {
                this.blades.push({
                    width: randomRange(6, 14),
                    length: this.height * randomRange(0.75, 1.15),
                    spread: randomRange(-25, 25),
                    swaySpeed: randomRange(0.6, 1.2),
                    swayAmp: randomRange(14, 28),
                    phaseOffset: randomRange(0, Math.PI * 2),
                    colorStart: type === "ribbon" ? "#064e3b" : "#065f46",
                    colorMid: type === "ribbon" ? "#047857" : "#059669",
                    colorTip: type === "ribbon" ? "#34d399" : "#10b981",
                });
            }
        }

        draw(ctx, time) {
            ctx.save();
            ctx.translate(this.x, this.y);

            const motionFactor = isReducedMotion ? 0.15 : 1.0;

            for (let i = 0; i < this.blades.length; i++) {
                const b = this.blades[i];
                const sway = Math.sin(time * b.swaySpeed + b.phaseOffset) * b.swayAmp * motionFactor;

                const grad = ctx.createLinearGradient(0, 0, sway, -b.length);
                grad.addColorStop(0, b.colorStart);
                grad.addColorStop(0.5, b.colorMid);
                grad.addColorStop(1, b.colorTip);

                ctx.fillStyle = grad;
                ctx.globalAlpha = 0.85;

                ctx.beginPath();
                ctx.moveTo(-b.width * 0.5, 0);

                // Bezier ribbon blade
                ctx.bezierCurveTo(
                    b.spread * 0.4, -b.length * 0.35,
                    b.spread * 0.8 + sway * 0.5, -b.length * 0.7,
                    b.spread + sway, -b.length
                );
                ctx.bezierCurveTo(
                    b.spread * 0.8 + sway * 0.5 + b.width * 0.3, -b.length * 0.7,
                    b.spread * 0.4 + b.width * 0.4, -b.length * 0.35,
                    b.width * 0.5, 0
                );
                ctx.closePath();
                ctx.fill();
            }

            ctx.restore();
        }
    }

    /**
     * Submerged Rock / Substrate cluster
     */
    class AquariumRock {
        constructor(x, y, radiusX, radiusY, shade) {
            this.x = x;
            this.y = y;
            this.rx = radiusX;
            this.ry = radiusY;
            this.shade = shade;
        }

        draw(ctx) {
            ctx.save();
            ctx.translate(this.x, this.y);

            const grad = ctx.createRadialGradient(-this.rx * 0.3, -this.ry * 0.3, this.rx * 0.1, 0, 0, this.rx);
            grad.addColorStop(0, "#334155");
            grad.addColorStop(0.65, this.shade);
            grad.addColorStop(1, "#090d16");

            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.ellipse(0, 0, this.rx, this.ry, 0, 0, Math.PI * 2);
            ctx.fill();

            // Subtle highlight rim
            ctx.strokeStyle = "rgba(148, 163, 184, 0.2)";
            ctx.lineWidth = 1.2;
            ctx.stroke();

            ctx.restore();
        }
    }

    /**
     * Aerator Bubble
     */
    class Bubble {
        constructor(isInitial = false) {
            this.reset(isInitial);
        }

        reset(isInitial = false) {
            // Pick an emitter column
            const emitterX = CONFIG.bubbleEmitters[Math.floor(Math.random() * CONFIG.bubbleEmitters.length)];
            this.x = emitterX * width + randomRange(-18, 18);
            this.y = isInitial ? randomRange(0, height) : height + randomRange(10, 40);
            this.radius = randomRange(1.8, 5.2);
            this.speed = randomRange(1.2, 2.6);
            this.wobblePhase = randomRange(0, Math.PI * 2);
            this.wobbleSpeed = randomRange(2.0, 4.0);
            this.wobbleAmp = randomRange(0.4, 1.2);
            this.alpha = randomRange(0.45, 0.85);
        }

        update(dt) {
            const motionFactor = isReducedMotion ? 0.3 : 1.0;
            this.y -= this.speed * motionFactor;
            this.wobblePhase += dt * this.wobbleSpeed * motionFactor;
            this.x += Math.sin(this.wobblePhase) * this.wobbleAmp * motionFactor;

            // Gentle expansion near surface
            if (this.y < height * 0.3) {
                this.radius += dt * 0.15;
            }

            // Surface pop / reset
            if (this.y < -10) {
                this.reset(false);
            }
        }

        draw(ctx) {
            ctx.save();
            ctx.globalAlpha = this.alpha;

            // Glassy bubble gradient
            const grad = ctx.createRadialGradient(
                this.x - this.radius * 0.35,
                this.y - this.radius * 0.35,
                this.radius * 0.1,
                this.x,
                this.y,
                this.radius
            );
            grad.addColorStop(0, "rgba(255, 255, 255, 0.9)");
            grad.addColorStop(0.4, "rgba(186, 230, 253, 0.5)");
            grad.addColorStop(0.85, "rgba(56, 189, 248, 0.2)");
            grad.addColorStop(1, "rgba(255, 255, 255, 0.8)");

            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fill();

            // Specular highlight crescent
            ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
            ctx.lineWidth = 0.6;
            ctx.beginPath();
            ctx.arc(this.x - this.radius * 0.2, this.y - this.radius * 0.2, this.radius * 0.55, -Math.PI * 0.7, -Math.PI * 0.1);
            ctx.stroke();

            ctx.restore();
        }
    }

    /**
     * Cursor Water Ripple
     */
    class WaterRipple {
        constructor(x, y) {
            this.x = x;
            this.y = y;
            this.radius = 8;
            this.maxRadius = randomRange(55, 90);
            this.alpha = 0.25;
            this.speed = randomRange(40, 65);
        }

        update(dt) {
            this.radius += this.speed * dt;
            this.alpha = 0.25 * (1 - this.radius / this.maxRadius);
        }

        draw(ctx) {
            if (this.alpha <= 0.01) return;
            ctx.save();
            ctx.strokeStyle = `rgba(186, 230, 253, ${this.alpha})`;
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.stroke();
            ctx.restore();
        }
    }

    // -------------------------------------------------------------------------
    // Setup & Initialization
    // -------------------------------------------------------------------------
    function init() {
        canvas = document.getElementById("aquarium-canvas");
        if (!canvas) {
            console.warn("ShinerAI Aquarium: Canvas element #aquarium-canvas not found.");
            return;
        }

        ctx = canvas.getContext("2d");
        if (!ctx) return;

        // Check accessibility reduced-motion
        const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
        isReducedMotion = motionQuery.matches;
        motionQuery.addEventListener("change", (e) => {
            isReducedMotion = e.matches;
        });

        // Tab visibility management to conserve battery/CPU
        document.addEventListener("visibilitychange", () => {
            isTabVisible = !document.hidden;
            if (isTabVisible && !animId) {
                lastTime = performance.now();
                animId = requestAnimationFrame(renderLoop);
            }
        });

        // Mouse listeners on window so canvas does not intercept pointer clicks
        window.addEventListener("mousemove", onMouseMove, { passive: true });
        window.addEventListener("mouseleave", onMouseLeave, { passive: true });
        window.addEventListener("touchstart", onTouchMove, { passive: true });
        window.addEventListener("touchmove", onTouchMove, { passive: true });
        window.addEventListener("touchend", onMouseLeave, { passive: true });

        // Viewport resize listener
        window.addEventListener("resize", onResize, { passive: true });

        // Initial setup
        onResize();
        lastTime = performance.now();
        animId = requestAnimationFrame(renderLoop);
    }

    function onResize() {
        if (!canvas) return;
        width = window.innerWidth;
        height = window.innerHeight;
        dpr = Math.min(window.devicePixelRatio || 1, 2);

        canvas.width = width * dpr;
        canvas.height = height * dpr;
        canvas.style.width = width + "px";
        canvas.style.height = height + "px";

        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        rebuildEnvironment();
    }

    function determineFishCount() {
        if (width > 1024) return CONFIG.fishCountDesktop;
        if (width > 640) return CONFIG.fishCountTablet;
        return CONFIG.fishCountMobile;
    }

    function rebuildEnvironment() {
        // 1. Create Groups
        groups = [];
        for (let i = 0; i < CONFIG.groupCount; i++) {
            const startX = randomRange(width * 0.15, width * 0.85);
            const startY = randomRange(height * 0.15, height * 0.85);
            groups.push(new FishGroup(i, startX, startY));
        }

        // 2. Create Goldfish
        const targetFish = determineFishCount();
        fishList = [];
        for (let i = 0; i < targetFish; i++) {
            const groupIdx = i % CONFIG.groupCount;
            const parentGroup = groups[groupIdx];
            const fish = new Goldfish(i, groupIdx, parentGroup.x, parentGroup.y);
            fishList.push(fish);
        }

        // Sort fish by depth z for accurate painter's algorithm rendering
        fishList.sort((a, b) => a.z - b.z);

        // 3. Create Aquatic Plants (Focused near edges & bottom corners)
        plants = [];
        // Left plant thicket (Vallisneria ribbons + broad leafs)
        plants.push(new AquaticPlant(width * 0.03, height, "ribbon", 0.7));
        plants.push(new AquaticPlant(width * 0.07, height, "broad", 0.45));
        plants.push(new AquaticPlant(width * 0.12, height, "ribbon", 0.55));

        // Right plant thicket
        plants.push(new AquaticPlant(width * 0.88, height, "ribbon", 0.6));
        plants.push(new AquaticPlant(width * 0.94, height, "broad", 0.5));
        plants.push(new AquaticPlant(width * 0.97, height, "ribbon", 0.72));

        // Background center plants (shorter, subtle)
        plants.push(new AquaticPlant(width * 0.35, height, "broad", 0.35));
        plants.push(new AquaticPlant(width * 0.65, height, "ribbon", 0.38));

        // 4. Create Substrate Rocks
        rocks = [];
        rocks.push(new AquariumRock(width * 0.06, height - 15, 65, 30, "#1e293b"));
        rocks.push(new AquariumRock(width * 0.14, height - 10, 48, 22, "#334155"));
        rocks.push(new AquariumRock(width * 0.86, height - 14, 55, 26, "#1e293b"));
        rocks.push(new AquariumRock(width * 0.95, height - 18, 70, 32, "#0f172a"));
        rocks.push(new AquariumRock(width * 0.50, height - 8, 40, 18, "#1e293b"));

        // 5. Create Aerator Bubbles
        const bubbleCount = width > 768 ? CONFIG.bubbleCountDesktop : CONFIG.bubbleCountMobile;
        bubbles = [];
        for (let i = 0; i < bubbleCount; i++) {
            bubbles.push(new Bubble(true));
        }

        waterRipples = [];
    }

    // -------------------------------------------------------------------------
    // Input Handling
    // -------------------------------------------------------------------------
    function onMouseMove(e) {
        mouse.x = e.clientX;
        mouse.y = e.clientY;
        mouse.active = true;
        mouse.lastMoved = performance.now();

        // Spawn very subtle water ripple when moving briskly
        if (!isReducedMotion && Math.random() < 0.18) {
            waterRipples.push(new WaterRipple(mouse.x, mouse.y));
            if (waterRipples.length > 8) waterRipples.shift();
        }
    }

    function onTouchMove(e) {
        if (e.touches && e.touches.length > 0) {
            mouse.x = e.touches[0].clientX;
            mouse.y = e.touches[0].clientY;
            mouse.active = true;
            mouse.lastMoved = performance.now();
        }
    }

    function onMouseLeave() {
        mouse.active = false;
        mouse.x = -9999;
        mouse.y = -9999;
    }

    // -------------------------------------------------------------------------
    // Environmental Rendering Functions
    // -------------------------------------------------------------------------

    /**
     * Draw deep teal/blue water gradient & underwater sunbeams
     */
    function drawWaterAndLighting(ctx, time) {
        // Deep aquatic water gradient
        const waterGrad = ctx.createLinearGradient(0, 0, 0, height);
        waterGrad.addColorStop(0, "#083344");   // Upper clear teal
        waterGrad.addColorStop(0.35, "#0e455f"); // Mid water body
        waterGrad.addColorStop(0.75, "#082f49"); // Lower water
        waterGrad.addColorStop(1, "#021626");   // Substrate floor depth

        ctx.fillStyle = waterGrad;
        ctx.fillRect(0, 0, width, height);

        // Volumetric Sunbeams streaming through water surface
        ctx.save();
        ctx.globalCompositeOperation = "screen";

        const beamAngle = 0.22; // Angled light shafts
        for (let i = 0; i < CONFIG.sunbeamCount; i++) {
            const beamX = width * (0.15 + i * 0.24);
            const beamWidth = width * 0.16;
            const beamSway = Math.sin(time * 0.4 + i * 1.5) * 20;

            const beamAlpha = isReducedMotion
                ? 0.04
                : 0.035 + Math.sin(time * 0.6 + i * 1.2) * 0.015;

            const sunbeamGrad = ctx.createLinearGradient(beamX, 0, beamX + height * Math.tan(beamAngle) + beamSway, height);
            sunbeamGrad.addColorStop(0, `rgba(186, 230, 253, ${beamAlpha * 1.8})`);
            sunbeamGrad.addColorStop(0.5, `rgba(125, 211, 252, ${beamAlpha})`);
            sunbeamGrad.addColorStop(1, "rgba(56, 189, 248, 0)");

            ctx.fillStyle = sunbeamGrad;
            ctx.beginPath();
            ctx.moveTo(beamX - beamWidth * 0.3, 0);
            ctx.lineTo(beamX + beamWidth * 0.7, 0);
            ctx.lineTo(beamX + height * Math.tan(beamAngle) + beamWidth + beamSway, height);
            ctx.lineTo(beamX + height * Math.tan(beamAngle) - beamWidth * 0.5 + beamSway, height);
            ctx.closePath();
            ctx.fill();
        }

        // Gentle surface caustics shimmer
        const causticAlpha = isReducedMotion ? 0.03 : 0.04 + Math.sin(time * 1.2) * 0.015;
        const causticGrad = ctx.createLinearGradient(0, 0, 0, height * 0.35);
        causticGrad.addColorStop(0, `rgba(224, 242, 254, ${causticAlpha * 1.5})`);
        causticGrad.addColorStop(1, "rgba(224, 242, 254, 0)");
        ctx.fillStyle = causticGrad;
        ctx.fillRect(0, 0, width, height * 0.35);

        ctx.restore();
    }

    /**
     * Draw gravel/substrate bed at the bottom
     */
    function drawSubstrate(ctx) {
        ctx.save();
        const bedHeight = 28;
        const bedGrad = ctx.createLinearGradient(0, height - bedHeight, 0, height);
        bedGrad.addColorStop(0, "rgba(15, 23, 42, 0)");
        bedGrad.addColorStop(0.4, "rgba(15, 23, 42, 0.75)");
        bedGrad.addColorStop(1, "rgba(2, 6, 23, 0.95)");

        ctx.fillStyle = bedGrad;
        ctx.fillRect(0, height - bedHeight, width, bedHeight);

        // Draw rocks
        for (let i = 0; i < rocks.length; i++) {
            rocks[i].draw(ctx);
        }
        ctx.restore();
    }

    // -------------------------------------------------------------------------
    // Main Animation Render Loop
    // -------------------------------------------------------------------------
    function renderLoop(timestamp) {
        if (!isTabVisible) {
            animId = null;
            return;
        }

        const dt = Math.min((timestamp - lastTime) / 1000, 0.1);
        lastTime = timestamp;
        simTime += dt;

        // Auto-deactivate mouse after inactivity
        if (mouse.active && performance.now() - mouse.lastMoved > 2500) {
            mouse.active = false;
        }

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // 1. Water gradient, sunbeams & lighting
        drawWaterAndLighting(ctx, simTime);

        // 2. Update Group Centroids
        for (let i = 0; i < groups.length; i++) {
            groups[i].update(dt);
        }

        // 3. Background Plants (Deep Layer)
        for (let i = 0; i < plants.length; i++) {
            if (i % 2 === 0) plants[i].draw(ctx, simTime);
        }

        // 4. Update & Draw Goldfish (Layered by depth z)
        for (let i = 0; i < fishList.length; i++) {
            const fish = fishList[i];
            fish.update(dt);
            fish.draw(ctx);
        }

        // 5. Foreground Plants (Crisp near edges)
        for (let i = 0; i < plants.length; i++) {
            if (i % 2 !== 0) plants[i].draw(ctx, simTime);
        }

        // 6. Substrate & Rocks
        drawSubstrate(ctx);

        // 7. Aerator Bubbles
        for (let i = 0; i < bubbles.length; i++) {
            bubbles[i].update(dt);
            bubbles[i].draw(ctx);
        }

        // 8. Water Cursor Ripples
        for (let i = waterRipples.length - 1; i >= 0; i--) {
            const rip = waterRipples[i];
            rip.update(dt);
            rip.draw(ctx);
            if (rip.alpha <= 0.01) {
                waterRipples.splice(i, 1);
            }
        }

        animId = requestAnimationFrame(renderLoop);
    }

    // Initialize when DOM content is ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
