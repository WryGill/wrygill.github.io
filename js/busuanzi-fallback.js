window.setTimeout(() => {
  for (const metric of ['site_uv', 'site_pv', 'page_pv']) {
    const counter = document.getElementById(`busuanzi_value_${metric}`);
    if (counter && !/^\d+$/.test(counter.textContent.trim())) {
      counter.textContent = '—';
    }
  }
}, 5000);
