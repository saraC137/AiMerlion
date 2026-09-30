// =================================================================
// 🧠 STATE
// =================================================================
const CANDIDATE_ID = {{ candidate_id }};
const RAW_TEXT = {{ raw_text | tojson }};
const SCHEMA = {{ schema_json | safe }};
const COLOR_MAP = {{ color_map_json | safe }};
let annotations = {{ annotations_json | safe }};
let activeEntityType = null;

// =================================================================
// 🎨 ENTITY PALETTE — Build the button grid
// =================================================================
function buildPalette() {
    const palette = document.getElementById('entityPalette');
    palette.innerHTML = '';

    const categories = {};
    for (const [name, info] of Object.entries(SCHEMA.entities)) {
        const cat = info.category;
        if (!categories[cat]) categories[cat] = [];
        categories[cat].push({name, ...info});
    }

    for (const [cat, entities] of Object.entries(categories)) {
        const catLabel = document.createElement('div');
        catLabel.style.cssText = 'font-size:0.62rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.5px; margin-top:8px; margin-bottom:4px;';
        catLabel.textContent = cat;
        palette.appendChild(catLabel);

        for (const e of entities) {
            const btn = document.createElement('button');
            btn.className = 'entity-btn';
            btn.dataset.entity = e.name;
            btn.style.background = e.color + '20';
            btn.style.color = e.color;
            btn.innerHTML = e.label + (e.shortcut_key ?
                ` <span class="key-hint">${e.shortcut_key}</span>` : '');
            btn.title = e.description.substring(0, 120) + '...';
            btn.onclick = () => selectEntity(e.name);
            palette.appendChild(btn);
        }
    }
}

function selectEntity(name) {
    activeEntityType = (activeEntityType === name) ? null : name;
    document.querySelectorAll('.entity-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.entity === activeEntityType);
    });
}

// =================================================================
// 📝 TEXT RENDERING — Display text with inline annotations
// =================================================================
function renderText() {
    const container = document.getElementById('textContent');

    // ── Separate positioned vs. manual/edited annotations ────────
    // Positioned: have valid char_start/char_end → render inline in RAW_TEXT
    // Manual/Edited: char_start < 0 → can't map to RAW_TEXT, show separately
    //
    // 🐛 BUG FIX: Previously ALL annotations went through
    // RAW_TEXT.substring(char_start, char_end), which meant:
    //   - Manual entries (char_start=-1) produced empty text
    //   - Edited entries still showed OLD text from the original position
    // Now manual/edited annotations render using their .text property! ✨
    const positioned = [];
    const manualEdited = [];

    for (const ann of annotations) {
        if (ann.char_start >= 0 && ann.char_end > ann.char_start) {
            positioned.push(ann);
        } else {
            manualEdited.push(ann);
        }
    }

    // Sort positioned annotations by start position
    const sorted = [...positioned].sort((a, b) => a.char_start - b.char_start);

    let html = '';
    let pos = 0;

    for (const ann of sorted) {
        // Text before this annotation
        if (ann.char_start > pos) {
            html += escapeHtml(RAW_TEXT.substring(pos, ann.char_start));
        }

        // The annotation span — uses RAW_TEXT slice for positioned annotations
        const color = COLOR_MAP[ann.entity_type] || '#888';
        const spanText = RAW_TEXT.substring(ann.char_start, ann.char_end);
        const label = SCHEMA.entities[ann.entity_type]?.label || ann.entity_type;
        const autoTag = ann.annotator?.startsWith('auto') ? ' 🤖' : '';

        html += `<span class="anno-span" `
            + `style="background:${color}20; border-color:${color}; color:${color};" `
            + `data-idx="${annotations.indexOf(ann)}" `
            + `onclick="focusAnnotation(${annotations.indexOf(ann)})" `
            + `title="${label}: ${escapeHtml(spanText)}">`
            + `<span class="anno-label" style="background:${color}; color:#000;">${label}${autoTag}</span>`
            + escapeHtml(spanText)
            + `</span>`;

        pos = ann.char_end;
    }

    // Remaining text
    if (pos < RAW_TEXT.length) {
        html += escapeHtml(RAW_TEXT.substring(pos));
    }

    // ── Manual / Edited annotations section ──────────────────────────
    // These are annotations created or edited via Save Preview that
    // don't have valid char positions in RAW_TEXT. We show them in a
    // dedicated block below the main text so nothing is invisible! 💅
    if (manualEdited.length > 0) {
        html += `<div style="margin-top:16px; padding-top:12px; border-top:1px dashed var(--border-color);">`;
        html += `<div style="font-size:0.72rem; color:var(--text-muted); margin-bottom:8px;">`;
        html += `✏️ Manually added/edited entries (${manualEdited.length}):`;
        html += `</div>`;
        for (const ann of manualEdited) {
            const color = COLOR_MAP[ann.entity_type] || '#888';
            const label = SCHEMA.entities[ann.entity_type]?.label || ann.entity_type;
            const displayText = ann.text || '(empty)';
            html += `<span class="anno-span" `
                + `style="background:${color}20; border-color:${color}; color:${color}; margin:2px 4px 2px 0; display:inline-block;" `
                + `data-idx="${annotations.indexOf(ann)}" `
                + `onclick="focusAnnotation(${annotations.indexOf(ann)})" `
                + `title="${label}: ${escapeHtml(displayText)}">`
                + `<span class="anno-label" style="background:${color}; color:#000;">${label} ✏️</span>`
                + escapeHtml(displayText)
                + `</span> `;
        }
        html += `</div>`;
    }

    container.innerHTML = html;
    updateAnnoList();
}

/**
 * escapeHtml(text)
 *
 * Escapes HTML special characters so text can be safely injected
 * into BOTH innerHTML AND attribute values (like value="...").
 *
 * 🐛 BUG FIX: The old version used textContent→innerHTML which
 * only escapes <, >, and &. It did NOT escape double quotes!
 * When annotation text contained " characters (e.g. "Big Corp"),
 * the preview form's value="..." attributes would BREAK, causing:
 *   - Input values getting truncated at the first quote
 *   - confirmSave() reading corrupted data
 *   - Silent JS errors preventing saveAnnotations() from running
 *
 * Think of it like writing your name on a form but the pen
 * explodes at the quotation marks — the rest is just a smear! 🖊️💥
 *
 * Now we also escape " → &quot; and ' → &#39; so the text is
 * safe EVERYWHERE in HTML — tags, attributes, you name it! 💅
 *
 * @param {string} text - Raw text to escape
 * @returns {string}    - HTML-safe escaped string
 */
function escapeHtml(text) {
    if (text == null) return '';
    const str = String(text);
    const div = document.createElement('div');
    div.textContent = str;
    // innerHTML escapes <, >, & — but NOT quotes!
    // We add quote escaping for safe attribute injection.
    return div.innerHTML
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// =================================================================
// ✏️ ANNOTATION CREATION — Select text to create annotation
// =================================================================
document.getElementById('textPanel').addEventListener('mouseup', (e) => {
    if (!activeEntityType) return;

    const selection = window.getSelection();
    if (!selection || selection.isCollapsed) return;

    const range = selection.getRangeAt(0);
    const container = document.getElementById('textContent');

    // ── Guard: if the selection boundary is inside an .anno-label chip
    //    (the floating coloured label above a span), the user accidentally
    //    dragged into a UI element that isn't raw text. Abort gracefully.
    const startEl = range.startContainer.parentElement;
    const endEl   = range.endContainer.parentElement;
    if (
        startEl?.classList.contains('anno-label') ||
        endEl?.classList.contains('anno-label')
    ) {
        selection.removeAllRanges();
        showToast('⚠️ Selection clipped a label chip — try again!', 'warning');
        return;
    }

    // ── Guard: selection must be fully inside the text content container
    if (!container.contains(range.commonAncestorContainer)) {
        selection.removeAllRanges();
        return;
    }

    // Calculate character offsets in the raw text
    // We need to map DOM positions back to raw text positions
    const charStart = getTextOffset(container, range.startContainer, range.startOffset);
    const charEnd   = getTextOffset(container, range.endContainer,   range.endOffset);

    

    if (charStart === null || charEnd === null || charStart >= charEnd) {
        selection.removeAllRanges();
        return;
    }

    const selectedText = RAW_TEXT.substring(charStart, charEnd).trim();
    if (!selectedText) {
        selection.removeAllRanges();
        return;
    }

    // Adjust charStart/charEnd inward to strip any leading/trailing whitespace
    // so the highlight hugs the actual word, not trailing spaces
    const rawSlice  = RAW_TEXT.substring(charStart, charEnd);
    const trimStart = charStart + (rawSlice.length - rawSlice.trimStart().length);
    const trimEnd   = charEnd   - (rawSlice.length - rawSlice.trimEnd().length);

    // ══════════════════════════════════════════════════════════════════
    // 🧬 NESTED ENTITY HANDLING — The Fabulous Layer Cake! 🎂
    // ══════════════════════════════════════════════════════════════════
    // Check if this span overlaps an existing annotation.
    // If the parent allows nesting for this entity type, use layer 1!
    // Otherwise, block the overlap as before.
    
    let targetLayer = 0;
    const overlappingParent = annotations.find(a =>
        a.layer === 0 &&
        !(trimEnd <= a.char_start || trimStart >= a.char_end)
    );
    
    if (overlappingParent) {
        // Check if parent entity allows this child to nest inside
        const parentDef = SCHEMA.entities[overlappingParent.entity_type];
        const allowsNesting = parentDef?.allows_nesting === true;
        const allowedChildren = parentDef?.nested_children || [];
        
        if (allowsNesting && allowedChildren.includes(activeEntityType)) {
            // ✅ Nesting allowed! Use layer 1 for the child annotation
            targetLayer = 1;
            console.log(`🧬 Nesting ${activeEntityType} inside ${overlappingParent.entity_type} on layer 1`);
        } else {
            // ❌ Nesting NOT allowed for this combination
            showToast(
                `⚠️ Cannot nest ${activeEntityType} inside ${overlappingParent.entity_type}. ` +
                `Delete the parent annotation first, or this entity type is not allowed as a child.`,
                'warning'
            );
            selection.removeAllRanges();
            return;
        }
    }

    // Create the new annotation object (now with smart layer!)
    annotations.push({
        entity_type: activeEntityType,
        char_start:  trimStart,
        char_end:    trimEnd,
        text:        RAW_TEXT.substring(trimStart, trimEnd),
        layer:       targetLayer,  // 🆕 Dynamic layer based on nesting!
        confidence:  1.0,
        annotator:   'human'
    });

    // Mark IAA annotations as unsaved whenever a new label is added in IAA mode
    if (iaaMode) setIAASaveStatus('unsaved');

    selection.removeAllRanges();
    renderText();
    showToast(`✅ Added ${activeEntityType}: "${selectedText.substring(0, 40)}${selectedText.length > 40 ? '...' : ''}"`);
});

/**
 * getTextOffset(root, node, offset)
 *
 * 🐛 BUG FIX — THE GREAT OFFSET DRAMA OF 2026 🎭
 *
 * The original version used a TreeWalker that walked ALL text nodes,
 * including the text inside floating .anno-label chips (e.g. "PERSON",
 * "EMAIL"). Those labels are pure UI decoration injected by renderText()
 * — they are NOT part of the raw resume text at all!
 *
 * So every existing annotation label was silently adding extra characters
 * to the offset count, causing subsequent selections to land in the WRONG
 * position. Like a choreographer counting the stagehands' steps instead
 * of just the dancers'! 💃🕺
 *
 * THE FIX: we pass a custom NodeFilter that tells the TreeWalker to SKIP
 * any text node whose parent is an .anno-label element.
 *
 * @param {Element}  root   - The #textContent container element
 * @param {Node}     node   - The DOM node where the selection boundary lives
 * @param {number}   offset - Character offset within that node
 * @returns {number|null}   - Character offset in RAW_TEXT, or null if not found
 */
function getTextOffset(root, node, offset) {

    // ── Custom filter: skip .anno-label text so we don't count
    //    UI labels (e.g. "PERSON", "EMAIL") as raw text characters.
    const skipLabels = {
        acceptNode(n) {
            // If the text node's parent is an .anno-label chip, skip it entirely
            if (n.parentElement && n.parentElement.classList.contains('anno-label')) {
                return NodeFilter.FILTER_SKIP;
            }
            return NodeFilter.FILTER_ACCEPT;
        }
    };

    // Walk only REAL text nodes (skipping label nodes via the filter above)
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, skipLabels);
    let charCount = 0;

    while (walker.nextNode()) {
        if (walker.currentNode === node) {
            // Found the exact text node — return accumulated count + local offset
            return charCount + offset;
        }
        charCount += walker.currentNode.textContent.length;
    }

    // ── Fallback: handles the rare case where the selection boundary lands
    //    on an ELEMENT node rather than a text node (e.g. user clicks at the
    //    very end of an .anno-span). We walk sibling elements and sum only
    //    the "real" text (via innerText of non-label children).
    if (node.nodeType !== Node.TEXT_NODE) {
        const walker2 = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, skipLabels);
        let count = 0;

        while (walker2.nextNode()) {
            // Check if this text node is inside the target element node
            if (node.contains && node.contains(walker2.currentNode)) {
                return count + offset;
            }
            count += walker2.currentNode.textContent.length;
        }
        // Last-resort: return accumulated count (best guess)
        return count;
    }

    return null;  // Could not resolve — selection will be ignored gracefully
}

// =================================================================
// 📋 ANNOTATION LIST — Sidebar list with delete buttons
// =================================================================
function updateAnnoList() {
    // annoList may be absent if the sidebar section has been removed —
    // null-guard so we never crash when writing to it. 💅
    const list = document.getElementById('annoList');

    // Update badge counts everywhere they appear
    const badge = document.getElementById('annoCountBadge');
    if (badge) badge.textContent = annotations.length;

    // Labels badge in toolbar (mirrors annotation count)
    const headerBadge = document.getElementById('headerLabelsBadge');
    if (headerBadge) headerBadge.textContent = annotations.length;

    // 📋 Annotations badge next to Edge button in toolbar
    const headerAnnotationsBadge = document.getElementById('headerAnnotationsBadge');
    if (headerAnnotationsBadge) headerAnnotationsBadge.textContent = annotations.length;

    // If the sidebar list element doesn't exist, nothing more to do here.
    if (!list) return;

    if (!annotations.length) {
        // Empty state message — friendly and helpful
        list.innerHTML = '<div class="anno-list-empty">No annotations yet.<br>Select a label above, then drag to highlight text! ✨</div>';
        return;
    }

    // Render each annotation as a list item with delete button
    list.innerHTML = annotations.map((a, i) => {
        const color = COLOR_MAP[a.entity_type] || '#888';
        const label = SCHEMA.entities[a.entity_type]?.label || a.entity_type;
        // Truncate long text for display — 50 chars max
        const text = (a.text || '').substring(0, 50);
        const autoIcon = a.annotator?.startsWith('auto') ? ' 🤖' : '';

        return `<div class="anno-list-item">
            <span class="anno-type" style="background:${color}20; color:${color};">${label}${autoIcon}</span>
            <span class="anno-text">"${escapeHtml(text)}"</span>
            <button class="delete-btn" onclick="deleteAnnotation(${i})" title="Delete">✕</button>
        </div>`;
    }).join('');
}

function deleteAnnotation(idx) {
    annotations.splice(idx, 1);
    // Mark IAA annotations as unsaved whenever a label is removed in IAA mode
    if (iaaMode) setIAASaveStatus('unsaved');
    renderText();
}

function focusAnnotation(idx) {
    /*
     * 🔦 FOCUS ANNOTATION
     *
     * Called when the user clicks a coloured span in the resume text.
     * We want to show them WHERE that annotation lives in the Labels list —
     * like tapping someone on the shoulder and pointing to their name on
     * the guest list! 🎉
     *
     * Flow:
     *   1. Open the Labels inspector modal (re-renders the full list)
     *   2. openLabelsModal(idx) handles the scroll + pulse highlight
     *      via requestAnimationFrame so the DOM is painted first.
     */
    openLabelsModal(idx);
}

function clearAll() {
    if (!confirm('Remove ALL annotations? This cannot be undone.')) return;
    annotations = [];
    renderText();
    showToast('🗑️ All annotations cleared');
}

