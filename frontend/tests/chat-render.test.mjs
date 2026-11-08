import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, writeFile, unlink } from 'node:fs/promises';
import { JSDOM } from 'jsdom';
import ts from 'typescript';
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';

test('answer renders when browser scrolling returns a Promise', async () => {
  const dom = new JSDOM('<div id="root"></div>', { url: 'http://localhost:3000' });
  const original = new Map();
  for (const [key, value] of Object.entries({window:dom.window,document:dom.window.document,
    navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,
    localStorage:dom.window.localStorage,IS_REACT_ACT_ENVIRONMENT:true})) {
    original.set(key, Object.getOwnPropertyDescriptor(globalThis,key));
    Object.defineProperty(globalThis,key,{configurable:true,writable:true,value});
  }
  let scrolls=0;
  dom.window.HTMLElement.prototype.scrollIntoView=function(){scrolls++;return Promise.resolve();};
  // React's old input polyfill is immaterial to this rendering regression.
  dom.window.HTMLElement.prototype.attachEvent=function(){};
  dom.window.HTMLElement.prototype.detachEvent=function(){};
  const oldFetch=globalThis.fetch;
  globalThis.fetch=async url=>({ok:true,json:async()=>String(url).endsWith('/health')
    ?{model:'local-test',local_only:true,scope:'smoke'}
    :{answer:'예약 가능한 장비는 20GP와 40GP입니다.',status:'answered',reason:'문서 근거',
      citations:[{book_id:'AS007',page:1,chunk_id:'chunk-1'}],contexts:[{book_id:'AS007',page:1,chunk_id:'chunk-1',text:'20GP와 40GP'}],
      request_id:'test-request',trace_id:null,mode:'hybrid_rerank',model:'local-test',latency_ms:10,
      input_tokens:10,output_tokens:10,warnings:[]}});
  const moduleFile=new URL('../.chat-render-test.mjs',import.meta.url);
  let root;
  try {
    const source=await readFile(new URL('../app/page.tsx',import.meta.url),'utf8');
    const code=ts.transpileModule(source,{compilerOptions:{jsx:ts.JsxEmit.ReactJSX,module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022}}).outputText;
    await writeFile(moduleFile,code);
    const {default:Home}=await import(moduleFile.href+'?test='+Date.now());
    const errors=[];
    root=createRoot(document.getElementById('root'),{onUncaughtError:error=>errors.push(error)});
    await act(async()=>{root.render(React.createElement(Home));});
    const suggestion=[...document.querySelectorAll('.suggestions button')][0];
    assert.ok(suggestion,'initial suggestions render');
    await act(async()=>{suggestion.click();});
    assert.match(document.querySelector('.message.assistant').textContent,/20GP와 40GP/);
    assert.match(document.querySelector('.citations').textContent,/AS007/);
    await act(async()=>{document.querySelector('.new-chat').click();});
    assert.ok(document.querySelector('.welcome'),'new chat survives effect cleanup');
    assert.ok(scrolls>1,'scroll effect runs across message changes');
    assert.deepEqual(errors,[]);
    await act(async()=>root.unmount());root=null;
  } finally {
    if(root)await act(async()=>root.unmount());
    globalThis.fetch=oldFetch;
    for(const [key,descriptor] of original){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}
    dom.window.close();
    await unlink(moduleFile).catch(()=>{});
  }
});
