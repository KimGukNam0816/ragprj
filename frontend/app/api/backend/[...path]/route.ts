import { NextRequest, NextResponse } from 'next/server';
export const runtime = 'nodejs';
export const maxDuration = 180;
async function proxy(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const target = path.join('/');
  if (!['query','status','health','connections','memory','evaluation/summary','evaluation/runs'].includes(target)) return NextResponse.json({detail:'Not found'},{status:404});
  // 평가 요약은 ?run= 으로 실행을 고르므로 쿼리 문자열을 그대로 넘긴다
  const search = target.startsWith('evaluation/') ? req.nextUrl.search : '';
  try {
    const response = await fetch(`${process.env.BACKEND_URL || 'http://127.0.0.1:8000'}/${target}${search}`, {
      method:req.method, headers:{'Content-Type':'application/json','X-Entrypoint':'web',
        // 토큰은 Next 서버에만 있고 브라우저로 내려가지 않는다.
        ...(process.env.BACKEND_ACCESS_TOKEN?{'X-Access-Token':process.env.BACKEND_ACCESS_TOKEN}:{})},
      body:req.method==='POST'?await req.text():undefined, cache:'no-store', signal:AbortSignal.timeout(180000),
    });
    return new NextResponse(await response.text(),{status:response.status,headers:{'Content-Type':'application/json'}});
  } catch { return NextResponse.json({detail:{message:'백엔드에 연결할 수 없습니다. FastAPI 서버 실행 상태를 확인해 주세요.'}},{status:503}); }
}
export const GET=proxy;
export const POST=proxy;
