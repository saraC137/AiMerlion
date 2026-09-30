// =================================================================
// 🗂️ TAB SWITCHING — Navigate between side panel sections
// =================================================================
/**
 * switchTab(tabId, clickedBtn)
 * 
 * LEGACY — kept for backward compat but now mostly a no-op.
 * The old tab system is replaced by inspector modals! 🎭
 */
function switchTab(tabId, clickedBtn) {
    // Redirect to the new modal system
    if (tabId === 'bio') openBIOModal();
    else if (tabId === 'edge') openEdgeModal();
    else if (tabId === 'annotations') openLabelsModal();
}

// =================================================================
// 🔍 INSPECTOR MODALS — Labels / BIO / Edge popups
// =================================================================

/** Currently open inspector (for Escape key routing) */
let activeInspector = null;

/**
 * openLabelsModal(focusIdx)
 * 
 * Opens a popup showing ALL annotations in a scrollable list.
 * Like unrolling the FULL guest list at the gala! 📜✨
 *
 * @param {number|null} focusIdx - Optional index of the annotation to
 *   scroll to and highlight. Pass null (or omit) to just open the list.
 */
function openLabelsModal(focusIdx = null) {
    const body = document.getElementById('labelsModalBody');
    const badge = document.getElementById('labelsModalBadge');
    if (badge) badge.textContent = annotations.length;

    if (!annotations.length) {
        body.innerHTML = '<div class="anno-list-empty">No annotations yet. Select a label above, then highlight text! ✨</div>';
    } else {
        // Render every annotation as a list row.
        // data-idx stamps the array index on the DOM so focusAnnotation()
        // can find the right row with a single querySelector. 🎯
        body.innerHTML = annotations.map((a, i) => {
            const color = COLOR_MAP[a.entity_type] || '#888';
            const label = SCHEMA.entities[a.entity_type]?.label || a.entity_type;
            const text = (a.text || '').substring(0, 80);
            const autoIcon = a.annotator?.startsWith('auto') ? ' 🤖' : ' 👤';
            return `<div class="anno-list-item" data-idx="${i}">
                <span class="anno-type" style="background:${color}20; color:${color};">${label}${autoIcon}</span>
                <span class="anno-text">"${escapeHtml(text)}"</span>
                <button class="delete-btn" onclick="deleteAnnotation(${i}); openLabelsModal();" title="Delete">✕</button>
            </div>`;
        }).join('');
    }

    // Open the inspector overlay first so the DOM is visible
    openInspector('labels');

    // If a specific annotation was requested, scroll to it and flash it.
    // requestAnimationFrame defers until after the browser has painted the
    // newly opened modal — without this, scrollIntoView is a no-op because
    // the element isn't visible yet. Timing is everything, darling! ⏱️✨
    if (focusIdx !== null) {
        requestAnimationFrame(() => {
            const target = body.querySelector(`[data-idx="${focusIdx}"]`);
            if (!target) return;

            // Remove any previous focus highlight before adding the new one
            // (handles the case where modal was already open) 💅
            body.querySelectorAll('.anno-list-item--focused')
                .forEach(el => el.classList.remove('anno-list-item--focused'));

            // Scroll the row into the centre of the modal body
            target.scrollIntoView({ behavior: 'smooth', block: 'center' });

            // Add the pulse class — CSS animation fires automatically
            target.classList.add('anno-list-item--focused');
        });
    }
}

/**
 * openBIOModal()
 * 
 * Opens the BIO tag preview popup and auto-loads data.
 * Behind the curtain at the stage machinery! ⚙️
 */
function openBIOModal() {
    openInspector('bio');
    // Auto-load if there are annotations
    if (annotations.length > 0) {
        loadBIOPreview();
    }
}

/**
 * openEdgeModal()
 * 
 * Opens the edge case inspector popup and auto-scans.
 * The divas of annotation need SPECIAL attention! 💅
 */
function openEdgeModal() {
    openInspector('edge');
    loadEdgeCases();
}

/**
 * openInspector(name)
 * Generic helper — opens the named inspector overlay.
 */
function openInspector(name) {
    // Close any already-open inspector first
    if (activeInspector) closeInspector(activeInspector);

    const overlay = document.getElementById(name + 'Overlay');
    if (overlay) overlay.classList.add('open');
    activeInspector = name;
    document.addEventListener('keydown', inspectorEscapeHandler);
}

/**
 * closeInspector(name)
 * Generic helper — closes the named inspector overlay.
 */
function closeInspector(name) {
    const overlay = document.getElementById(name + 'Overlay');
    if (overlay) overlay.classList.remove('open');
    activeInspector = null;
    document.removeEventListener('keydown', inspectorEscapeHandler);
}

/**
 * inspectorEscapeHandler(event)
 * Escape key closes the active inspector modal.
 */
function inspectorEscapeHandler(event) {
    if (event.key === 'Escape' && activeInspector) {
        event.preventDefault();
        closeInspector(activeInspector);
    }
}

