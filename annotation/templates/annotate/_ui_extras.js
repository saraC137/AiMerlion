// =================================================================
// 🚪 PANEL RESIZE — Drag handle to resize the side panel
// =================================================================
/**
 * How it works, darling! 💅
 *
 * The .anno-layout grid uses `grid-template-columns: 1fr 5px var(--side-width)`.
 * When the user drags the resize handle, we calculate how far the mouse is
 * from the RIGHT edge of the viewport — that distance becomes the new
 * --side-width value, which the grid picks up instantly via CSS variables.
 *
 * Think of it as sliding a partition wall in a nightclub: the dance floor
 * (text panel) gets bigger or smaller as the VIP room (side panel) moves! 🪩
 *
 * Min/max are enforced in CSS (.side-panel min-width/max-width) but we also
 * clamp here in JS so the CSS variable never goes to a nonsensical value.
 */
(function initPanelResize() {
    const handle    = document.getElementById('resizeHandle');
    const layout    = document.getElementById('annoLayout');
    const MIN_WIDTH = 220;   /* px — narrowest usable panel */
    const MAX_WIDTH = 600;   /* px — widest panel before it hogs the screen */

    if (!handle || !layout) return;  // Guard: elements must exist

    let dragging = false;

    handle.addEventListener('mousedown', (e) => {
        e.preventDefault();          // Prevent text selection during drag
        dragging = true;
        handle.classList.add('dragging');
        // Use document listeners so drag works even if mouse leaves the handle
        document.addEventListener('mousemove', onDrag);
        document.addEventListener('mouseup', stopDrag);
    });

    function onDrag(e) {
        if (!dragging) return;

        // Distance from the mouse to the right edge = desired panel width
        const newWidth = Math.min(
            MAX_WIDTH,
            Math.max(MIN_WIDTH, window.innerWidth - e.clientX)
        );

        // Update the CSS variable — the grid reflows automatically 🎉
        layout.style.setProperty('--side-width', newWidth + 'px');
    }

    function stopDrag() {
        dragging = false;
        handle.classList.remove('dragging');
        document.removeEventListener('mousemove', onDrag);
        document.removeEventListener('mouseup', stopDrag);
    }
})();

// =================================================================
// ⌨️ KEYBOARD SHORTCUTS
// =================================================================
document.addEventListener('keydown', (e) => {
    // Ignore if typing in an input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    // Number keys 1-9, H → select entity type
    const entities = Object.entries(SCHEMA.entities);
    for (const [name, info] of entities) {
        if (info.shortcut_key && e.key === info.shortcut_key) {
            e.preventDefault();
            selectEntity(name);
            return;
        }
    }

    // Delete/Backspace → delete last annotation
    if (e.key === 'Delete' || e.key === 'Backspace') {
        if (annotations.length > 0 && !e.target.closest('.anno-span')) {
            e.preventDefault();
            annotations.pop();
            renderText();
            showToast('🗑️ Deleted last annotation');
        }
    }

    // Ctrl+S → direct save as draft (no preview needed)
    if (e.ctrlKey && !e.shiftKey && e.key === 's') {
        e.preventDefault();
        saveAnnotations('in_progress');
    }

    // Ctrl+Shift+S → save AND validate via preview (for Mark Complete)
    if (e.ctrlKey && e.shiftKey && e.key === 'S') {
        e.preventDefault();
        showSavePreview('completed');
    }

    // Escape → deselect entity type
    if (e.key === 'Escape') {
        activeEntityType = null;
        document.querySelectorAll('.entity-btn').forEach(b => b.classList.remove('active'));
    }
});

// =================================================================
// 🍞 TOAST NOTIFICATION
// =================================================================
/**
 * showToast(msg, type, duration)
 *
 * 💅 The glow-up that plain browser alert() always deserved!
 * Displays a beautiful animated toast notification in the bottom-right
 * corner. Uses the #toast-container div + CSS from the stylesheet.
 *
 * @param {string} msg      - Message text to display
 * @param {string} type     - 'success' | 'warning' | 'error' | 'info'
 * @param {number} duration - Auto-dismiss in ms (default: 4000)
 */
function showToast(msg, type = 'success', duration = 4000) {
    const container = document.getElementById('toast-container');
    if (!container) return;

    // Map type to an appropriate icon
    const icons = { success: '✅', warning: '⚠️', error: '❌', info: 'ℹ️' };
    const icon = icons[type] || '✅';

    // Build the toast element using the CSS classes from our stylesheet
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <span class="toast-icon">${icon}</span>
        <div class="toast-content">
            <div class="toast-msg" style="color:var(--text-primary);font-size:0.83rem;line-height:1.5;">
                ${msg.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
            </div>
        </div>
        <button class="toast-close" onclick="this.closest('.toast')._dismiss()" title="Dismiss">✕</button>
        <div class="toast-progress" style="animation-duration:${duration}ms;"></div>
    `;

    // Attach dismiss helper directly to element for clean self-reference
    toast._dismiss = () => {
        if (toast._dismissed) return;
        toast._dismissed = true;
        clearTimeout(toast._timer);
        toast.classList.add('toast-hiding');
        toast.addEventListener('animationend', () => toast.remove(), { once: true });
    };

    container.appendChild(toast);

    // Auto-dismiss after duration
    toast._timer = setTimeout(() => toast._dismiss(), duration);
}

/**
 * showDraftSavedCelebration()
 * 
 * The cute little sister to the big completion party!
 * Quick, fabulous, and gone before you blink — perfect for draft saves.
 * Like the makeup artist giving you a mirror check backstage: "Slay, queen!"
 */
function showDraftSavedCelebration() {
    const overlay = document.createElement('div');
    overlay.style.cssText = `
        position:fixed; inset:0; pointer-events:none; z-index:9999;
        display:flex; align-items:center; justify-content:center;
        animation: draftFade 2s ease forwards;
    `;

    // Tiny confetti (just 8 pieces, soft pastel colors)
    let confetti = '';
    const colors = ['#7ee8fa', '#eeb8ff', '#56d364', '#f778ba', '#e3b341'];
    for (let i = 0; i < 8; i++) {
        const c = colors[i % colors.length];
        const left = 30 + Math.random() * 40;
        confetti += `<div style="position:absolute; width:8px; height:8px; border-radius:50%; 
            background:${c}; left:${left}%; top:-20px; 
            animation: miniConfetti ${1.2 + Math.random()}s linear forwards;"></div>`;
    }

    overlay.innerHTML = `
        <div style="background:rgba(14,17,23,0.95); border:2px solid #7ee8fa; border-radius:9999px;
            padding:14px 32px; display:flex; align-items:center; gap:12px; box-shadow:0 0 30px #7ee8fa80;">
            ${confetti}
            <span style="font-size:2rem;">💾</span>
            <div>
                <div style="color:#7ee8fa; font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700;">
                    DRAFT SAVED, BITCH! ✨
                </div>
                <div style="color:#8b949e; font-size:0.82rem;">Your work is locked & loaded, honey</div>
            </div>
        </div>
    `;

    document.body.appendChild(overlay);

    // Auto-dismiss + click anywhere to dismiss
    setTimeout(() => overlay.remove(), 2200);
    overlay.addEventListener('click', () => overlay.remove(), {once: true});
}

// Add these keyframes once (put this inside your <style> or right after the other @keyframes)
const draftStyle = document.createElement('style');
draftStyle.textContent = `
    @keyframes draftFade { 0% {opacity:1; transform:scale(0.9)} 80% {opacity:1} 100% {opacity:0; transform:scale(1.05)} }
    @keyframes miniConfetti { to { transform:translateY(180px) rotate(720deg); opacity:0; } }
`;
document.head.appendChild(draftStyle);

/**
 * showCompletionCelebration()
 *
 * 🎊 THE GRAND FINALE — Full-screen celebration when marking a candidate Complete!
 * Auto-dismisses in 3.5s OR on any click/Escape keypress.
 * Like the final curtain call — lights, confetti, APPLAUSE! 🎭✨
 */
function showCompletionCelebration() {
    // Remove any stale overlay first (defensive coding, darling! 💅)
    const existing = document.getElementById('completionCelebration');
    if (existing) existing.remove();

    // Inject CSS keyframes just once
    if (!document.getElementById('celebStyles')) {
        const style = document.createElement('style');
        style.id = 'celebStyles';
        style.textContent = `
            @keyframes celebFadeIn  { from{opacity:0} to{opacity:1} }
            @keyframes celebFadeOut { from{opacity:1;transform:scale(1)} to{opacity:0;transform:scale(1.04)} }
            @keyframes celebCardIn  { from{transform:translateY(-30px) scale(0.9);opacity:0} to{transform:translateY(0) scale(1);opacity:1} }
            @keyframes celebPulse   { 0%,100%{transform:scale(1)} 50%{transform:scale(1.08)} }
            @keyframes confettiFall { 0%{transform:translateY(-20px) rotate(0deg);opacity:1} 100%{transform:translateY(200px) rotate(720deg);opacity:0} }
        `;
        document.head.appendChild(style);
    }

    // Build the backdrop overlay
    const overlay = document.createElement('div');
    overlay.id = 'completionCelebration';
    overlay.style.cssText = `
        position:fixed; inset:0; background:rgba(0,0,0,0.82);
        z-index:9998; display:flex; align-items:center; justify-content:center;
        animation:celebFadeIn 0.3s ease forwards; cursor:pointer;
        backdrop-filter:blur(6px);
    `;

    // Generate confetti dots
    const colors = ['#56d364','#7ee8fa','#eeb8ff','#e3b341','#f778ba'];
    let confettiHTML = '';
    for (let i = 0; i < 14; i++) {
        const c = colors[i % colors.length];
        const sz = 6 + Math.floor(Math.random() * 8);
        const left = 4 + Math.floor(Math.random() * 92);
        const delay = (Math.random() * 1.8).toFixed(2);
        const dur = (1.4 + Math.random() * 1.2).toFixed(2);
        confettiHTML += `<div style="position:absolute;width:${sz}px;height:${sz}px;
            border-radius:50%;background:${c};left:${left}%;top:-12px;
            animation:confettiFall ${dur}s ${delay}s ease-in forwards;"></div>`;
    }

    // Build the card
    const card = document.createElement('div');
    card.style.cssText = `
        background: linear-gradient(135deg, #0e1117 0%, #1a2332 50%, #0e1117 100%);
        border: 1px solid rgba(86,211,100,0.45);
        border-radius: 20px; padding: 48px 56px; text-align: center;
        max-width: 440px; position: relative; overflow: hidden;
        box-shadow: 0 0 60px rgba(86,211,100,0.2), 0 24px 80px rgba(0,0,0,0.6);
        animation: celebCardIn 0.4s cubic-bezier(0.34,1.56,0.64,1) forwards;
    `;
    card.innerHTML = `
        ${confettiHTML}
        <div style="font-size:3.8rem;margin-bottom:14px;animation:celebPulse 0.9s ease infinite;">🏆</div>
        <div style="font-family:'Syne',system-ui;font-size:1.6rem;font-weight:800;
            color:#56d364;letter-spacing:-0.5px;margin-bottom:10px;">
            Annotation Complete!
        </div>
        <div style="font-size:0.95rem;color:#8b949e;line-height:1.7;margin-bottom:22px;">
            This candidate is officially<br>
            <strong style="color:#e6edf3;">runway-ready</strong>
            and queued for export! ✨
        </div>
        <div style="display:flex;gap:10px;justify-content:center;
            font-size:0.72rem;color:#545d68;flex-wrap:wrap;">
            <span>✅ Saved to database</span>
            <span>·</span>
            <span>🎯 Status: Completed</span>
            <span>·</span>
            <span>📤 Ready to export</span>
        </div>
        <div style="margin-top:18px;font-size:0.65rem;color:#3a4250;">
            Click anywhere or press Escape to close
        </div>
    `;

    overlay.appendChild(card);
    document.body.appendChild(overlay);

    // Dismiss logic — click, Escape, or 3.5s auto-dismiss
    const dismiss = () => {
        if (!overlay.parentNode) return;
        overlay.style.animation = 'celebFadeOut 0.25s ease forwards';
        overlay.addEventListener('animationend', () => overlay.remove(), { once: true });
        document.removeEventListener('keydown', escHandler);
    };
    const escHandler = (e) => { if (e.key === 'Escape') dismiss(); };
    overlay.addEventListener('click', dismiss);
    document.addEventListener('keydown', escHandler);
    setTimeout(dismiss, 3500);
}

// =================================================================
// ❓ HELP MODAL — Open / Close + Shortcut Grid Builder
// =================================================================

/**
 * openHelp()
 * Opens the help manual modal by adding the 'open' class to the overlay.
 * Also traps Escape key to close it, like a civilised exit! 💅
 */
function openHelp() {
    document.getElementById('helpOverlay').classList.add('open');
    // Listen for Escape key to close the modal
    document.addEventListener('keydown', closeHelpOnEscape);
}

/**
 * closeHelp()
 * Hides the help modal and removes the Escape key listener.
 */
function closeHelp() {
    document.getElementById('helpOverlay').classList.remove('open');
    document.removeEventListener('keydown', closeHelpOnEscape);
}

/**
 * closeHelpOnBackdrop(event)
 * Closes the modal only if the user clicked the dark backdrop itself,
 * NOT if they clicked inside the white modal card.
 * Like leaving the party only when you WANT to — not when someone bumps you! 🎉
 *
 * @param {MouseEvent} event - The click event on the overlay div
 */
function closeHelpOnBackdrop(event) {
    // event.target is the element that was actually clicked.
    // If it's the overlay div (not the modal card inside), close.
    if (event.target === document.getElementById('helpOverlay')) {
        closeHelp();
    }
}

/**
 * closeHelpOnEscape(event)
 * Keyboard handler: closes the modal when Escape is pressed.
 * Added/removed dynamically so it doesn't interfere with annotation shortcuts
 * when the modal is closed.
 *
 * @param {KeyboardEvent} event
 */
function closeHelpOnEscape(event) {
    if (event.key === 'Escape') {
        closeHelp();
    }
}

/**
 * buildShortcutGrid()
 *
 * Reads the entity schema and injects each entity's keyboard shortcut
 * into the Help Modal shortcut grid — so the manual is always in sync
 * with the actual schema, no matter how many entities get added! 🎹
 *
 * Think of it as a self-updating setlist for our show. ✨
 */
function buildShortcutGrid() {
    const grid = document.getElementById('shortcutGrid');
    if (!grid) return;  // Modal might not be in the DOM yet — guard gracefully

    // Loop through the schema and add one grid item per entity with a shortcut
    for (const [name, info] of Object.entries(SCHEMA.entities)) {
        if (!info.shortcut_key) continue;  // Skip entities without shortcuts

        const item = document.createElement('div');
        item.className = 'shortcut-item';
        // Color the label to match the entity's colour in the palette
        item.innerHTML = `
            <span class="kbd">${info.shortcut_key}</span>
            <span style="color:${info.color}; font-weight:600;">${info.label}</span>
        `;
        grid.appendChild(item);
    }
}

