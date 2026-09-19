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

const code = () => editor ? editor.getValue() : ($('editor').querySelector('textarea') || {}).value || '';

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
  impl.template = s.template;
  $('cases').innerHTML = impl.cases.map((c, i) =>
    `<span class="case ${i === 0 ? 'on' : ''}" data-case="${esc(c)}">${esc(c)}</span>`).join('');
  $('cases').querySelectorAll('[data-case]').forEach(el =>
    el.onclick = () => { if (!busy) el.classList.toggle('on'); });
}

const chosenCases = () => [...$('cases').querySelectorAll('.case.on')].map(el => el.dataset.case);

async function save() {
  if (!impl || !impl.ready) return;
  const r = await api('/api/save', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                    body: JSON.stringify({kernel: impl.name, code: code()})});
  $('saved').textContent = r.ok ? '已保存' : '保存失败:' + r.data.error;
}

// ---------------------------------------------------------------- 跑

function setBusy(v) {
  busy = v;
  $('b-run').disabled = v; $('b-submit').disabled = v; $('impl').disabled = v; $('target').disabled = v;
}

async function go(mode) {
  if (busy || !impl || !impl.ready) return;
  setBusy(true);
  $('verdict').textContent = mode === 'check' ? '运行中…' : '判题中…';
  $('verdict').className = 'verdict busy';
  $('log').textContent = '';
  let buf = '';
  try {
    const res = await fetch('/api/run', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({kernel: impl.name, target: $('target').value, mode,
                            cases: chosenCases(), code: code()}),
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

function renderSolutions() {
  const p = S.problems.find(p => p.name === problem);
  $('pane-solutions').innerHTML =
    `<h2>同一道题的其他实现</h2><p>同样的数学、同样的 case 与容差,换一种 DSL 重写一遍,数字可以直接横着比。</p>`
    + p.impls.map(k => `<div class="row" data-impl="${esc(k.name)}">
         <span class="k">${esc(k.toolchain)}</span>
         <span class="badge ${k.name === impl.name ? 'cur' : ''}">${esc(k.name)}</span>
         <span class="m">${k.ready ? esc(k.features.join(', ') || '无特性门禁') : esc(k.note)}</span>
       </div>`).join('');
  $('pane-solutions').querySelectorAll('[data-impl]').forEach(el =>
    el.onclick = () => { if (!busy) selectImpl(el.dataset.impl).then(renderSolutions); });
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
  if (busy || !impl || !impl.template) return alert('这道题的 ' + (impl ? impl.toolchain : '') + ' 还没有骨架模板');
  if (confirm('用骨架覆盖当前代码?')) setCode(impl.template, impl.language);
};
window.addEventListener('keydown', e => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); go(e.shiftKey ? 'run' : 'check'); }
});

initEditor().then(load);
