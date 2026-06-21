(function(){
  const storageKey='dashboardSidebarCollapsed';
  const root=document.documentElement;

  try{
    if(localStorage.getItem(storageKey)==='true') root.classList.add('sidebar-collapsed');
  }catch(e){}

  function wrapLinkLabels(sidebar){
    sidebar.querySelectorAll('.global-nav-link,.nav-link').forEach(link=>{
      if(link.querySelector('.sidebar-label')) return;
      const textNode=[...link.childNodes].find(node=>node.nodeType===Node.TEXT_NODE && node.textContent.trim());
      if(!textNode) return;
      const label=document.createElement('span');
      label.className='sidebar-label';
      label.textContent=textNode.textContent.trim();
      link.replaceChild(label,textNode);
      if(!link.title) link.title=label.textContent;
    });
  }

  function updateButton(button){
    const collapsed=root.classList.contains('sidebar-collapsed');
    button.setAttribute('aria-expanded',String(!collapsed));
    button.setAttribute('aria-label',collapsed?'Expand sidebar':'Collapse sidebar');
    button.title=collapsed?'Expand sidebar':'Collapse sidebar';
  }

  function init(){
    const sidebar=document.querySelector('.global-sidebar,.sidebar');
    if(!sidebar || sidebar.querySelector('.sidebar-toggle')) return;
    wrapLinkLabels(sidebar);

    const button=document.createElement('button');
    button.type='button';
    button.className='sidebar-toggle';
    button.innerHTML='<span></span><span></span><span></span>';
    button.addEventListener('click',()=>{
      root.classList.toggle('sidebar-collapsed');
      try{localStorage.setItem(storageKey,String(root.classList.contains('sidebar-collapsed')));}catch(e){}
      updateButton(button);
    });
    updateButton(button);
    sidebar.appendChild(button);
  }

  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',init,{once:true});
  else init();
})();
