/* kernellab 面板。左边题面,右边 Monaco 写算子,Run = check,Submit = check+bench+ncu。 */
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

let S = null;            // /api/state
let problem = null;      // 当前题名
let impl = null;         // 当前实现(= specs/<名>,决定语言与工具链)
let editor = null;       // Monaco 实例
let busy = false;
let saveTimer = null;

// ---------------------------------------------------------------- Monaco

const CUDA_KEYWORDS = ['__global__','__device__','__host__','__shared__','__constant__','__restrict__',
  '__syncthreads','__syncwarp','threadIdx','blockIdx','blockDim','gridDim','warpSize','atomicAdd',
  '__ldg','__expf','half','half2','nv_bfloat16','cudaStream_t','dim3'];

/* 片段补全:写 kernel 时真正常用的那些 API,带参数占位。Monaco 自带词法补全,这里补语义常用项。 */
const SNIPPETS = {
  python: [
    ['tl.program_id',  'tl.program_id(axis=${1:0})', '本 program 在 grid 第几格'],
    ['tl.arange',      'tl.arange(0, ${1:BLOCK})', '编译期常量长度的下标向量'],
    ['tl.load',        'tl.load(${1:ptrs}, mask=${2:mask}, other=${3:0.0})', '带掩码的块加载'],
    ['tl.store',       'tl.store(${1:ptrs}, ${2:val}, mask=${3:mask})', '带掩码的块写回'],
    ['tl.dot',         'tl.dot(${1:a}, ${2:b})', '块乘块,降到 tensor core'],
    ['tl.zeros',       'tl.zeros((${1:BLOCK_M}, ${2:BLOCK_N}), dtype=tl.float32)', 'fp32 累加器'],
    ['tl.max',         'tl.max(${1:x}, axis=${2:0})', '沿轴取最大'],
    ['tl.sum',         'tl.sum(${1:x}, axis=${2:0})', '沿轴求和'],
    ['tl.exp',         'tl.exp(${1:x})', '逐元素 exp'],
    ['tl.where',       'tl.where(${1:cond}, ${2:a}, ${3:b})', '逐元素选择'],
    ['tl.cdiv',        'tl.cdiv(${1:x}, ${2:y})', '向上取整除'],
    ['tl.constexpr',   'tl.constexpr', '编译期常量参数标注'],
    ['triton.jit',     '@triton.jit\ndef ${1:kernel}(${2:args}):\n    ${3:pass}', 'Triton kernel 骨架'],
    ['triton.cdiv',    'triton.cdiv(${1:x}, ${2:y})', '主机侧向上取整除'],
    ['num_warps',      'num_warps=${1:4}', '一个 CTA 几个 warp'],
    ['num_stages',     'num_stages=${1:3}', '共享内存流水级数(cp.async 多级缓冲)'],
  ],
  cpp: [
    ['__global__',     '__global__ void ${1:kernel}(${2:args}) {\n    ${3}\n}', 'kernel 骨架'],
    ['__shared__',     '__shared__ ${1:float} ${2:smem}[${3:size}];', '共享内存数组'],
    ['__syncthreads',  '__syncthreads();', 'block 内屏障'],
    ['tid',            'int ${1:tid} = blockIdx.x * blockDim.x + threadIdx.x;', '全局线程号'],
    ['guard',          'if (${1:i} < ${2:n}) {\n    ${3}\n}', '边界保护'],
    ['float4',         'float4 ${1:v} = reinterpret_cast<const float4*>(${2:ptr})[${3:i}];', '向量化加载'],
    ['cp.async',       'asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" :: "r"(${1:dst}), "l"(${2:src}));', 'Ampere 异步拷贝'],
  ],
};

function registerCompletions(monaco) {
  for (const [lang, items] of Object.entries(SNIPPETS)) {
    monaco.languages.registerCompletionItemProvider(lang, {
      provideCompletionItems(model, pos) {
        const w = model.getWordUntilPosition(pos);
        const range = {startLineNumber: pos.lineNumber, endLineNumber: pos.lineNumber,
                       startColumn: w.startColumn, endColumn: w.endColumn};
        return {suggestions: items.map(([label, snippet, doc]) => ({
          label, detail: doc, documentation: doc, insertText: snippet, range,
          kind: monaco.languages.CompletionItemKind.Snippet,
          insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
        }))};
      },
    });
  }
  // cpp 语法表不认 CUDA 关键字,补进去才会着色
  const cpp = monaco.languages.getLanguages().find(l => l.id === 'cpp');
  if (cpp && cpp.loader) cpp.loader().then(m => {
    if (m.language && m.language.keywords) m.language.keywords.push(...CUDA_KEYWORDS);
  }).catch(() => {});
}

function initEditor() {
  return new Promise(resolve => {
    if (typeof require === 'undefined') return resolve(false);   // 没 vendor 到 Monaco
    require.config({paths: {vs: '/static/vendor/monaco/vs'}});
    require(['vs/editor/editor.main'], () => {
      registerCompletions(monaco);
      editor = monaco.editor.create($('editor'), {
        value: '', language: 'python', theme: 'vs',
        fontSize: 13.5, fontFamily: 'ui-monospace, "SF Mono", Menlo, Consolas, monospace',
        minimap: {enabled: false}, scrollBeyondLastLine: false, automaticLayout: true,
        tabSize: 4, renderWhitespace: 'selection', padding: {top: 10},
      });
      editor.onDidChangeModelContent(() => {
        $('saved').textContent = '未保存';
        clearTimeout(saveTimer);
        saveTimer = setTimeout(save, 900);
      });
      editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => go('check'));
      editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.Enter, () => go('run'));
      resolve(true);
    });
  });
}

/* 拿编辑器里的代码。拿不到时返回 null 而不是空串 —— 空串会被当成「用户清空了文件」写回磁盘,
   等于一次静默的数据丢失。调用方见到 null 就别保存。 */
function code() {
  if (editor) return editor.getValue();
  const ta = $('editor').querySelector('textarea');
  return ta ? ta.value : null;
}

function setCode(text, language) {
  if (editor) {
    monaco.editor.setModelLanguage(editor.getModel(), language);
    editor.setValue(text);
  } else {                                   // Monaco 没取到时的退路:纯 textarea,功能不缺
    $('editor').innerHTML = '<textarea style="width:100%;height:100%;border:0;outline:0;padding:10px;'
      + 'font:13px ui-monospace,Menlo,monospace;resize:none"></textarea>';
    $('editor').querySelector('textarea').value = text;
  }
}

// ---------------------------------------------------------------- 数据

async function api(path, opts) {
  const r = await fetch(path, opts);
  return {ok: r.ok, status: r.status, data: await r.json()};
}

async function load() {
  S = (await api('/api/state')).data;
  $('repo').textContent = S.repo;
  $('target').innerHTML = S.targets.map(t =>
    `<option value="${t.name}">${t.name}${t.gpu ? ' · ' + t.gpu : ''}</option>`).join('');
  $('target').value = S.default_target;
  if (!problem) {
    const p = S.problems.find(p => p.ready) || S.problems[0];
    if (p) await selectProblem(p.name);
  } else {
    renderSubs();
  }
}

async function selectProblem(name) {
  problem = name;
  const p = S.problems.find(p => p.name === name);
  const d = (await api('/api/problem?name=' + encodeURIComponent(name))).data;
  $('pane-desc').innerHTML = d.description;
  $('pane-editorial').innerHTML = d.editorial;
  $('impl').innerHTML = p.impls.map(k =>
    `<option value="${k.name}" ${k.ready ? '' : 'disabled'}>${k.toolchain}${k.ready ? '' : '(缺源码)'}</option>`).join('');
  const first = p.impls.find(k => k.ready) || p.impls[0];
  await selectImpl(first.name);
  renderSolutions();
  renderSubs();
  showTab('desc');
}

async function selectImpl(name) {
  solIdx = null;                       // 换语言:阶梯长度不同,回列表页
  const p = S.problems.find(p => p.name === problem);
  impl = p.impls.find(k => k.name === name);
  $('impl').value = name;
  if (!impl.ready) {
    setCode('# ' + impl.note, 'python');
    $('srcpath').textContent = impl.note;
    $('cases').innerHTML = '';
    return;
  }
  const s = (await api('/api/source?kernel=' + encodeURIComponent(name))).data;
  setCode(s.code, s.language);
  $('srcpath').textContent = s.path;
  $('saved').textContent = '已保存';
  impl.backbone = s.backbone;
  $('cases').innerHTML = impl.cases.map((c, i) =>
    `<span class="case ${i === 0 ? 'on' : ''}" data-case="${esc(c)}">${esc(c)}</span>`).join('');
  $('cases').querySelectorAll('[data-case]').forEach(el =>
    el.onclick = () => { if (!busy) el.classList.toggle('on'); });
}

const chosenCases = () => [...$('cases').querySelectorAll('.case.on')].map(el => el.dataset.case);

async function save() {
  if (!impl || !impl.ready) return;
  const c = code();
  if (c === null) { $('saved').textContent = '编辑器没就绪,没保存'; return; }
  const r = await api('/api/save', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                    body: JSON.stringify({kernel: impl.name, code: c})});
  $('saved').textContent = r.ok ? '已保存' : '保存失败:' + r.data.error;
}

// ---------------------------------------------------------------- 跑

function setBusy(v) {
  busy = v;
  $('b-run').disabled = v; $('b-submit').disabled = v; $('impl').disabled = v; $('target').disabled = v;
}

async function go(mode) {
  if (busy || !impl || !impl.ready) return;
  const c = code();
  if (c === null) { alert('编辑器还没就绪,稍等一下再试'); return; }
  setBusy(true);
  $('verdict').textContent = mode === 'check' ? '运行中…' : '判题中…';
  $('verdict').className = 'verdict busy';
  $('log').textContent = '';
  let buf = '';
  try {
    const res = await fetch('/api/run', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({kernel: impl.name, target: $('target').value, mode,
                            cases: chosenCases(), code: c}),
    });
    $('saved').textContent = '已保存';
    if (!res.ok) {
      const e = await res.json();
      $('log').textContent = '✗ ' + e.error;
      $('verdict').textContent = '未能运行'; $('verdict').className = 'verdict bad';
      return;
    }
    const reader = res.body.getReader(), dec = new TextDecoder();
    for (;;) {
      const {done, value} = await reader.read();
      if (done) break;
      const chunk = dec.decode(value, {stream: true});
      buf += chunk;
      $('log').textContent += chunk;
      $('log').scrollTop = $('log').scrollHeight;
    }
    $('log').textContent = $('log').textContent.replace(/\n__KLAB_RUN__ \S+\n?/, '');

    const failed = /FAIL/.test(buf), refused = /\[requires\]/.test(buf);
    const compiled = !/Traceback|error:/i.test(buf);
    if (refused)      { $('verdict').textContent = '架构不满足'; $('verdict').className = 'verdict bad'; }
    else if (failed)  { $('verdict').textContent = '答案错误';   $('verdict').className = 'verdict bad'; }
    else if (!compiled) { $('verdict').textContent = '编译/运行错误'; $('verdict').className = 'verdict bad'; }
    else              { $('verdict').textContent = mode === 'check' ? '通过' : '已通过 · 见结果';
                        $('verdict').className = 'verdict ok'; }

    const m = buf.match(/__KLAB_RUN__ (\S+)/);
    await load();
    if (mode !== 'check' && m && m[1] !== '-' && !failed) await showReport(m[1], true);
  } catch (e) {
    $('log').textContent += '\n✗ ' + e;
    $('verdict').textContent = '中断'; $('verdict').className = 'verdict bad';
  } finally {
    setBusy(false);
  }
}

// ---------------------------------------------------------------- 左侧各 tab

function showTab(name) {
  document.querySelectorAll('#left-tabs button').forEach(b => b.classList.toggle('on', b.dataset.tab === name));
  document.querySelectorAll('.tabpane').forEach(p => p.classList.toggle('on', p.id === 'pane-' + name));
}

async function showReport(run, jump) {
  const r = await api('/api/report?run=' + encodeURIComponent(run));
  $('pane-result').innerHTML = r.ok ? r.data.html : `<p class="empty">${esc(r.data.error)}(${esc(run)})</p>`;
  if (jump) showTab('result');
}

/* 参考答案是「左边读、右边自己写」的读物,不是一点就把编辑器覆盖掉的跳板。
   所以:列表点进去是详情页(一次一级),代码默认藏在「看答案」后面,
   逼自己先照着讲解写一遍;要对照上一级改了哪一处,有 diff 视图。 */
let sols = [];            // 当前语言的阶梯
let solIdx = null;        // 打开第几级;null = 停在列表页
let solShow = false;      // 这一级的代码揭晓了没有(换级就重置)
let solView = 'code';     // 'code' | 'diff'
let diffEd = null;        // Monaco diff 实例,换视图/换级前要 dispose,否则模型泄漏

function dropDiff() {
  if (!diffEd) return;
  const m = diffEd.getModel();
  diffEd.dispose();
  if (m) { m.original.dispose(); m.modified.dispose(); }
  diffEd = null;
}

async function renderSolutions() {
  dropDiff();
  const p = S.problems.find(p => p.name === problem);
  const langs = p.impls.map(k => `<option value="${esc(k.name)}" ${k.name === impl.name ? 'selected' : ''}
      ${k.ready ? '' : 'disabled'}>${esc(k.toolchain)}</option>`).join('');
  const r = await api('/api/solutions?kernel=' + encodeURIComponent(impl.name));
  sols = r.ok ? r.data.solutions : [];
  if (solIdx !== null && solIdx >= sols.length) solIdx = null;
  if (solIdx === null) solList(langs); else solDetail();
}

function solList(langs) {
  $('pane-solutions').innerHTML =
    `<h2>参考答案</h2>
     <p>从零到最优的完整晋升路径,每一级只改一处。语言
       <select id="sol-lang">${langs}</select>
       —— 同样的数学、同样的 case 与容差,数字可以直接横着比。</p>
     <p class="hint">点进去先读讲解,代码藏在「看答案」后面 —— 建议先照着思路自己在右边写一遍再对。</p>`
    + (sols.length ? sols.map((s, i) => `
        <div class="row" data-open="${i}">
          <span class="badge cur">${esc(s.id.split('-')[0])}</span>
          <span class="k">${esc(s.title)}</span>
          <span class="n">读这一级 →</span>
        </div>`).join('')
       : `<p class="empty">这一语言还没有写参考答案(放在 problems/${esc(problem)}/solutions/&lt;工具链&gt;/)。</p>`);
  const sel = $('sol-lang');
  if (sel) sel.onchange = () => { if (!busy) { solIdx = null; selectImpl(sel.value).then(renderSolutions); } };
  $('pane-solutions').querySelectorAll('[data-open]').forEach(el => el.onclick = () => {
    openSol(+el.dataset.open);
  });
}

function openSol(i) {
  solIdx = i; solShow = false; solView = 'code';
  solDetail();
  $('pane-solutions').parentElement.scrollTop = 0;    // .body 才是滚动容器
}

function solDetail() {
  dropDiff();
  const s = sols[solIdx];
  $('pane-solutions').innerHTML = `
    <div class="solnav">
      <button class="ghost tiny" id="sol-back">← 阶梯</button>
      <span class="hint">${esc(impl.toolchain)} · 第 ${solIdx + 1} / ${sols.length} 级</span>
      <div class="spacer"></div>
      <button class="ghost tiny" id="sol-prev" ${solIdx === 0 ? 'disabled' : ''}>← 上一级</button>
      <button class="ghost tiny" id="sol-next" ${solIdx === sols.length - 1 ? 'disabled' : ''}>下一级 →</button>
    </div>
    <h2><span class="badge cur">${esc(s.id.split('-')[0])}</span> ${esc(s.title)}</h2>
    <pre class="note">${s.note ? esc(s.note) : '<i>这一级还没写讲解。</i>'}</pre>
    <div id="sol-answer"></div>`;
  $('sol-back').onclick = () => { solIdx = null; renderSolutions(); };
  $('sol-prev').onclick = () => openSol(solIdx - 1);
  $('sol-next').onclick = () => openSol(solIdx + 1);
  solAnswer();
}

function solAnswer() {
  dropDiff();
  const s = sols[solIdx], prev = sols[solIdx - 1];
  const box = $('sol-answer');
  if (!solShow) {
    box.innerHTML = `<button class="reveal" id="sol-reveal">👁 看答案</button>
      <p class="hint">先照着上面的思路在右边写一遍;卡住了、或者写完想对一下,再点开。</p>`;
    $('sol-reveal').onclick = () => { solShow = true; solAnswer(); };
    return;
  }
  box.innerHTML = `
    <div class="solbar">
      <button class="seg ${solView === 'code' ? 'on' : ''}" data-view="code">完整代码</button>
      <button class="seg ${solView === 'diff' ? 'on' : ''}" data-view="diff"
              ${prev ? `title="和「${esc(prev.title)}」比"` : 'disabled title="这是第 0 级,没有上一级"'}>与上一级的差异</button>
      <div class="spacer"></div>
      <button class="ghost tiny" id="sol-load" title="覆盖右边的编辑器">载入编辑器</button>
    </div>
    ${solView === 'diff' && prev ? `<p class="hint">和「${esc(prev.title)}」比,只比代码 —— 开头那段讲解每级都重写,比了全是噪音。</p>` : ''}
    <div id="sol-body"></div>`;
  box.querySelectorAll('[data-view]').forEach(b => b.onclick = () => {
    if (!b.disabled) { solView = b.dataset.view; solAnswer(); }
  });
  $('sol-load').onclick = () => {
    if (busy) return;
    if (confirm(`把「${s.title}」载入右边的编辑器?你现在写的代码会被覆盖。`)) {
      setCode(s.code, impl.language); save();
    }
  };
  const body = $('sol-body');
  if (typeof monaco === 'undefined') {           // 没 vendor 到 Monaco 时的退路
    body.innerHTML = `<pre class="codeview">${esc(s.code)}</pre>`;
    return;
  }
  if (solView === 'diff' && prev) {
    body.innerHTML = '<div class="diffbox" id="sol-diff"></div>';
    diffEd = monaco.editor.createDiffEditor($('sol-diff'), {
      readOnly: true, renderSideBySide: false, automaticLayout: true, theme: 'vs',
      fontSize: 12.5, fontFamily: 'ui-monospace, "SF Mono", Menlo, Consolas, monospace',
      minimap: {enabled: false}, scrollBeyondLastLine: false, renderOverviewRuler: false,
    });
    // 比的是 body 不是 code:每一级都会重写开头的讲解,连着比的话满屏都是散文
    diffEd.setModel({
      original: monaco.editor.createModel(prev.body, impl.language),
      modified: monaco.editor.createModel(s.body, impl.language),
    });
    return;
  }
  monaco.editor.colorize(s.code, impl.language, {tabSize: 4})
    .then(html => { body.innerHTML = `<pre class="codeview">${html}</pre>`; })
    .catch(() => { body.innerHTML = `<pre class="codeview">${esc(s.code)}</pre>`; });
}

function renderSubs() {
  const p = S.problems.find(p => p.name === problem);
  if (!p) return;
  const names = new Set(p.impls.map(k => k.name));
  const rows = S.runs.filter(r => names.has(r.kernel));
  $('pane-subs').innerHTML = '<h2>提交记录</h2>' + (rows.length ? rows.map(r => `
    <div class="row" data-run="${esc(r.id)}">
      <span class="badge ${r.ok ? 'ok' : 'bad'}">${r.ok ? '通过' : '失败'}</span>
      <span class="k">${esc(r.kernel)}</span>
      <span class="m">${esc(r.mode)} · ${esc(r.target)} · ${esc(r.when.slice(4, 8))} ${esc(r.when.slice(9, 11))}:${esc(r.when.slice(11, 13))}</span>
      <span class="n">${r.speedup ? r.speedup.toFixed(2) + '× torch' : ''}
        ${r.tflops ? '<br>' + r.tflops.toFixed(1) + ' TFLOPS' : ''}</span>
    </div>`).join('') : '<p class="empty">还没有提交</p>');
  $('pane-subs').querySelectorAll('[data-run]').forEach(el => el.onclick = async () => {
    const id = el.dataset.run;
    const r = S.runs.find(x => x.id === id);
    if (r.has_report) await showReport(id, true);
    if (r.has_code) {
      const s = await api('/api/submission?run=' + encodeURIComponent(id));
      if (s.ok) openSheet(`<h2>这次提交的代码</h2><p class="empty">${esc(id)}</p>`
        + Object.entries(s.data.files).map(([n, c]) =>
            `<p class="path">${esc(n)}</p><pre class="code"><code>${esc(c)}</code></pre>`).join(''));
    }
  });
}

function openSheet(html) {
  $('sheetbox').innerHTML = html;
  $('sheetbox').className = 'sheetbox tabpane on';
  $('sheet').hidden = false;
}
$('sheet').onclick = e => { if (e.target === $('sheet')) $('sheet').hidden = true; };

$('problem-menu').onclick = () => openSheet('<h2>题目列表</h2>' + S.problems.map(p => `
  <div class="row" data-prob="${esc(p.name)}">
    <span class="k">${esc(p.title)}</span>
    <span class="badge ${p.name === problem ? 'cur' : ''}">${p.impls.length} 种语言</span>
    <span class="m">${p.has_doc ? '' : '缺题面'}</span>
  </div>`).join('') + (S.problems.length ? '' : '<p class="empty">specs/ 下还没有接线</p>'));

document.addEventListener('click', e => {
  const el = e.target.closest('[data-prob]');
  if (el) { $('sheet').hidden = true; selectProblem(el.dataset.prob); }
});

// ---------------------------------------------------------------- 分栏拖拽

function drag(gutter, apply) {
  gutter.onmousedown = e => {
    e.preventDefault();
    const move = ev => apply(ev);
    const up = () => { document.removeEventListener('mousemove', move); document.removeEventListener('mouseup', up); };
    document.addEventListener('mousemove', move);
    document.addEventListener('mouseup', up);
  };
}
drag($('gutter-x'), e => {
  const r = $('main').getBoundingClientRect();
  $('left').style.width = Math.min(Math.max(e.clientX - r.left, 280), r.width - 360) + 'px';
});
drag($('gutter-y'), e => {
  const r = $('right').getBoundingClientRect();
  $('console-panel').style.height = Math.min(Math.max(r.bottom - e.clientY, 60), r.height - 140) + 'px';
});

// ---------------------------------------------------------------- 接线

document.querySelectorAll('#left-tabs button').forEach(b => b.onclick = () => showTab(b.dataset.tab));
$('b-run').onclick = () => go('check');
$('b-submit').onclick = () => go('run');
$('impl').onchange = () => selectImpl($('impl').value).then(renderSolutions);
$('b-reset').onclick = () => {
  if (busy || !impl || !impl.backbone) return alert('这道题的 ' + (impl ? impl.toolchain : '') + ' 还没有骨架');
  if (confirm('用骨架覆盖当前代码?函数体会清空,从头写。')) { setCode(impl.backbone, impl.language); save(); }
};
window.addEventListener('keydown', e => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); go(e.shiftKey ? 'run' : 'check'); }
});

initEditor().then(load);
