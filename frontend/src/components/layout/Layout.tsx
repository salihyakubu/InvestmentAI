import type { ReactNode } from 'react';
import Sidebar from './Sidebar';
import Header from './Header';

interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-gray-900">
      <Sidebar />
      <Header />
      <main className="ml-60 mt-16 p-6">{children}</main>
    </div>
  );
}
