import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = { title: '항로 · 해운 문서 도우미', description: '문서 근거와 함께 답하는 로컬 AI 해운 도우미' };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="ko"><body>{children}</body></html>;
}
