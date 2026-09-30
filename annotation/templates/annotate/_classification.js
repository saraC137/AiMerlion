// =================================================================
// 🧠 FUNCTION/INDUSTRY/SKILLS/TAGS CLASSIFICATION
// =================================================================

// Hardcoded options from the schema
const FUNCTION_OPTIONS = [
    "Administration", "Accounting & Finance", "Customer Service",
    "Engineering", "Facilities Management", "Fresh Graduates (Deg/Dip)",
    "Fresh Graduates (ITE)", "Human Resources", "IT", "Legal",
    "Management", "Marketing", "Procurement", "Production",
    "Sales", "Shipping & Logistics", "others"
];
const INDUSTRY_OPTIONS = [
    "Automotive", "Banking & Finance", "Building/Construction",
    "Chemicals", "Education", "Engineering - Aerospace",
    "Engineering - Precision", "Energy (Oil & Gas)", "Environment & Water",
    "Healthcare", "Hospitality & Tourism", "F&B", "FMCG",
    "Infocomm", "Medical Technology", "Marine & Offshore",
    "Retail", "Pharmaceutical/Biotech", "Robotics", "Semiconductor",
    "Supply Chain Mgt & Logistics", "Telco", "Trading", "Others"
];

// 🆕 In-memory arrays for skills & tags (synced with chip UI)
let currentHardSkills = [];
let currentSoftSkills = [];
let currentTags = [];

/**
 * renderSkillChips(containerId, items, chipClass, arrayRef)
 *
 * Renders an array of skill/tag strings as clickable chip elements
 * inside the given container. Each chip has a ✕ button to remove it.
 *
 * Think of it like pinning nametags on a board — each one removable
 * with a single click! 📌✨
 *
 * @param {string} containerId - DOM id of the container div
 * @param {string[]} items - Array of skill/tag strings
 * @param {string} chipClass - CSS class for chip styling ('hard'|'soft'|'ai-tag')
 * @param {string} arrayName - Which array to modify ('hard'|'soft'|'tag')
 */
function renderSkillChips(containerId, items, chipClass, arrayName) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';

    items.forEach((item, idx) => {
        const chip = document.createElement('span');
        chip.className = `skill-tag-chip ${chipClass}`;
        chip.innerHTML = `${escapeHtml(item)}<span class="tag-remove" onclick="removeSkillTag('${arrayName}', ${idx})">✕</span>`;
        container.appendChild(chip);
    });
}

/**
 * addSkillTag(type)
 *
 * Reads the corresponding input field, adds the value to the
 * in-memory array, re-renders chips, and clears the input.
 * Deduplicates automatically — no double outfits, darling! 👗
 *
 * @param {string} type - 'hard' | 'soft' | 'tag'
 */
function addSkillTag(type) {
    let input, arr, containerId, chipClass;

    if (type === 'hard') {
        input = document.getElementById('hardSkillInput');
        arr = currentHardSkills;
        containerId = 'hardSkillsContainer';
        chipClass = '';
    } else if (type === 'soft') {
        input = document.getElementById('softSkillInput');
        arr = currentSoftSkills;
        containerId = 'softSkillsContainer';
        chipClass = 'soft';
    } else {
        input = document.getElementById('tagInput');
        arr = currentTags;
        containerId = 'tagsContainer';
        chipClass = 'ai-tag';
    }

    if (!input) return;
    const val = input.value.trim();
    if (!val) return;

    // Deduplicate (case-insensitive check)
    if (!arr.some(s => s.toLowerCase() === val.toLowerCase())) {
        arr.push(val);
    }

    renderSkillChips(containerId, arr, chipClass, type);
    input.value = '';
    input.focus();
}

/**
 * removeSkillTag(type, index)
 *
 * Removes a skill/tag from the in-memory array by index and re-renders.
 * Like plucking a bad sequin off a gown — quick and painless! ✂️✨
 *
 * @param {string} type - 'hard' | 'soft' | 'tag'
 * @param {number} index - Array index to remove
 */
function removeSkillTag(type, index) {
    if (type === 'hard') {
        currentHardSkills.splice(index, 1);
        renderSkillChips('hardSkillsContainer', currentHardSkills, '', 'hard');
    } else if (type === 'soft') {
        currentSoftSkills.splice(index, 1);
        renderSkillChips('softSkillsContainer', currentSoftSkills, 'soft', 'soft');
    } else {
        currentTags.splice(index, 1);
        renderSkillChips('tagsContainer', currentTags, 'ai-tag', 'tag');
        syncPaletteState();
    }
}

// =================================================================
// ⚡ QUICK-PICK TAG PALETTE
// =================================================================

// Core tags grouped by category — mirrors ResumeClassifier.CORE_TAGS
const CORE_TAGS = {
    "Availability":  ["Available Now", "Immediate", "1-Month Notice", "2-Month Notice", "3-Month Notice"],
    "Work Mode":     ["Remote Only", "Hybrid OK", "On-site Only", "Open to Relocation", "Willing to Travel"],
    "Work Type":     ["Full-time", "Part-time", "Contract", "Freelance", "Temp"],
    "Seniority":     ["Fresh Grad", "Junior", "Mid-level", "Senior", "Lead", "Manager", "Director", "C-Suite"],
    "People":        ["Team Lead", "People Manager", "Individual Contributor", "Career Switch", "Return to Work"],
    "Languages":     ["English", "Mandarin", "Malay", "Tamil", "Bilingual", "Trilingual"],
    "Status":        ["Citizen (SG)", "PR (SG)", "EP Holder", "Citizen (MY)", "PR (MY)", "Visa Required"],
    "Tech":          ["Full Stack", "Frontend", "Backend", "Mobile Dev", "DevOps", "Data Science", "AI/ML", "Cybersecurity", "Cloud", "QA / Testing"],
    "Finance":       ["ACCA", "CPA", "CFA", "Big 4", "Audit", "Tax", "Payroll", "Financial Reporting", "SAP User"],
    "HR":            ["Generalist", "Talent Acquisition", "L&D", "Payroll", "HRIS", "Business Partner"],
    "Sales & Mktg":  ["B2B", "B2C", "SaaS Sales", "Key Account Mgmt", "Digital Mktg", "SEO/SEM", "CRM"],
};

const GROUP_COLORS = {
    "Availability": "#e3b341", "Work Mode": "#7ee8fa", "Work Type": "#7ee8fa",
    "Seniority": "#eeb8ff",    "People": "#eeb8ff",    "Languages": "#f778ba",
    "Status": "#f778ba",       "Tech": "#56d364",      "Finance": "#7ee8fa",
    "HR": "#eeb8ff",           "Sales & Mktg": "#e3b341",
};

/**
 * buildTagPalette()
 * Renders the quick-pick palette groups into #tagPaletteGroups.
 * Called once on classify modal open.
 */
function buildTagPalette() {
    const container = document.getElementById('tagPaletteGroups');
    if (!container || container.dataset.built) return;
    container.dataset.built = '1';

    for (const [group, chips] of Object.entries(CORE_TAGS)) {
        const color = GROUP_COLORS[group] || 'var(--accent-green)';

        const label = document.createElement('div');
        label.className = 'palette-group-label';
        label.textContent = group;
        label.style.color = color;
        container.appendChild(label);

        const row = document.createElement('div');
        row.className = 'palette-chips';

        chips.forEach(tag => {
            const btn = document.createElement('span');
            btn.className = 'palette-chip';
            btn.textContent = tag;
            btn.dataset.tag = tag;
            btn.style.borderColor = color + '55';
            btn.addEventListener('click', () => togglePaletteTag(tag, btn, color));
            row.appendChild(btn);
        });

        container.appendChild(row);
    }
}

/**
 * togglePaletteTag(tag, btn, color)
 * Adds or removes a tag when its palette chip is clicked.
 * Syncs the chip's visual state with currentTags.
 */
function togglePaletteTag(tag, btn, color) {
    const idx = currentTags.findIndex(t => t.toLowerCase() === tag.toLowerCase());
    if (idx === -1) {
        // Add
        currentTags.push(tag);
        btn.classList.add('active');
        btn.style.borderColor = color;
        btn.style.color = color;
    } else {
        // Remove
        currentTags.splice(idx, 1);
        btn.classList.remove('active');
        btn.style.borderColor = color + '55';
        btn.style.color = '';
    }
    renderSkillChips('tagsContainer', currentTags, 'ai-tag', 'tag');
    syncPaletteState();  // Keep all chips in sync
}

/**
 * syncPaletteState()
 * Marks palette chips as active/inactive based on currentTags.
 * Called after external changes (populate, remove chip).
 */
function syncPaletteState() {
    const palette = document.getElementById('tagPaletteGroups');
    if (!palette) return;
    palette.querySelectorAll('.palette-chip').forEach(btn => {
        const tag = btn.dataset.tag;
        const color = btn.style.borderColor.replace('55', '') || 'var(--accent-green)';
        const active = currentTags.some(t => t.toLowerCase() === tag.toLowerCase());
        btn.classList.toggle('active', active);
    });
}

/**
 * toggleTagPalette()
 * Shows/hides the quick-pick palette and builds it on first open.
 */
function toggleTagPalette() {
    const palette = document.getElementById('tagPalette');
    const btn = document.getElementById('paletteToggleBtn');
    if (!palette) return;
    const isOpen = palette.style.display !== 'none';
    palette.style.display = isOpen ? 'none' : 'block';
    btn.style.opacity = isOpen ? '0.75' : '1';
    btn.style.borderColor = isOpen ? '' : 'var(--accent-green)';
    if (!isOpen) {
        buildTagPalette();   // Build on first open
        syncPaletteState();  // Reflect any already-added tags
    }
}

/**
 * populateClassification(functionVal, industryVal, hardSkills, softSkills, tags)
 *
 * Populates the Function & Industry dropdowns AND the skill/tag chip
 * containers with the given values. Called on page load and after
 * re-classification.
 *
 * 🆕 Now handles the full classification suite: Function, Industry,
 *     Hard Skills, Soft Skills, and AI-generated Tags!
 */
function populateClassification(functionVal, industryVal, hardSkills, softSkills, tags) {
    const funcSelect = document.getElementById('functionSelect');
    const indSelect = document.getElementById('industrySelect');

    // Guard: if the Classify tab elements aren't in the DOM yet, bail out
    if (!funcSelect || !indSelect) {
        console.warn('⚠️ Classification selects not found in DOM');
        return;
    }

    // Normalize empty/null/undefined to empty string for safe comparison
    functionVal = (functionVal || '').trim();
    industryVal = (industryVal || '').trim();

    console.log('🧠 Populating classification:', {functionVal, industryVal,
        hardSkills: (hardSkills || []).length,
        softSkills: (softSkills || []).length,
        tags: (tags || []).length
    });

    funcSelect.innerHTML = FUNCTION_OPTIONS.map(f =>
        `<option value="${f}" ${f === functionVal ? 'selected' : ''}>${f}</option>`
    ).join('');
    indSelect.innerHTML = INDUSTRY_OPTIONS.map(i =>
        `<option value="${i}" ${i === industryVal ? 'selected' : ''}>${i}</option>`
    ).join('');

    // 🆕 Populate skill/tag chips
    currentHardSkills = Array.isArray(hardSkills) ? [...hardSkills] : [];
    currentSoftSkills = Array.isArray(softSkills) ? [...softSkills] : [];
    currentTags = Array.isArray(tags) ? [...tags] : [];

    renderSkillChips('hardSkillsContainer', currentHardSkills, '', 'hard');
    renderSkillChips('softSkillsContainer', currentSoftSkills, 'soft', 'soft');
    renderSkillChips('tagsContainer', currentTags, 'ai-tag', 'tag');
}

/**
 * openClassifyModal()
 *
 * Opens the Classification & Skills popup modal.
 * Uses the same inspector-overlay pattern as Labels/BIO/Edge.
 * Think of it as opening the talent dossier! 📋✨
 */
function openClassifyModal() {
    openInspector('classify');
}

function reclassify() {
    document.getElementById('classifyStatus').textContent = '⏳ Classifying...';

    // 🐛 BUG FIX: Send CURRENT in-memory annotations via POST!
    // Previously this was a GET that loaded from the DATABASE (stale data).
    // Now we send the live working annotations so the classifier sees
    // the latest SKILL and SOFT_SKILL entities — even if unsaved! 💅
    fetch(`/api/classify/${CANDIDATE_ID}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ annotations: annotations })
    })
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                document.getElementById('classifyStatus').textContent = '❌ ' + data.error;
                return;
            }
            populateClassification(
                data.Function, data.Industry,
                data.HardSkills || [], data.SoftSkills || [], data.Tags || []
            );
            document.getElementById('classifyStatus').textContent =
                `✅ Updated! ${(data.HardSkills||[]).length} hard, ${(data.SoftSkills||[]).length} soft, ${(data.Tags||[]).length} tags`;
        })
        .catch(err => {
            document.getElementById('classifyStatus').textContent = '❌ Error: ' + err;
        });
}

function saveClassification() {
    const functionVal = document.getElementById('functionSelect').value;
    const industryVal = document.getElementById('industrySelect').value;
    fetch('/api/save-classification', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            candidate_id: CANDIDATE_ID,
            function: functionVal,
            industry: industryVal,
            hard_skills: currentHardSkills,
            soft_skills: currentSoftSkills,
            tags: currentTags
        })
    })
    .then(r => r.json())
    .then(d => {
        if (d.success) {
            document.getElementById('classifyStatus').textContent = '✅ Saved!';
            showToast(
                '🧠 Classification Saved! Function, Industry, Skills & Tags locked in! 💼',
                'success',
                3500
            );
            // Auto-clear inline status after 3s so it does not go stale
            setTimeout(function() {
                var el = document.getElementById('classifyStatus');
                if (el && el.textContent === '✅ Saved!') { el.textContent = ''; }
            }, 3000);
        } else {
            document.getElementById('classifyStatus').textContent = '❌ Save failed';
            showToast('❌ Classification save failed — try again, darling!', 'error');
        }
    })
    .catch(function(err) {
        document.getElementById('classifyStatus').textContent = '❌ Network error';
        showToast('❌ Network error: ' + err, 'error');
    });
}

