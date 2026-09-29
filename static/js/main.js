// Click-to-reveal for blurred Tier-5 memes.
document.addEventListener('click', (ev) => {
  const btn = ev.target.closest('.sensitive .reveal');
  if (btn) btn.parentElement.classList.add('shown');
});
