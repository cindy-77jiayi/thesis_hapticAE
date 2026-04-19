import { Link } from "react-router";
import { CheckCircle2, AlertCircle, Bell, RefreshCw, ChevronRight } from "lucide-react";

export function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white pt-12 px-6 pb-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">Flow Demos</h1>
        <p className="text-sm text-gray-500">Explore interaction patterns</p>
      </div>

      <div className="space-y-3">
        <Link to="/success">
          <div className="bg-white rounded-2xl p-5 shadow-sm border border-gray-100 active:scale-[0.98] transition-transform">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center flex-shrink-0">
                <CheckCircle2 className="w-6 h-6 text-green-600" />
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="font-semibold text-gray-900 mb-0.5">Success Flow</h3>
                <p className="text-xs text-gray-500">Slide to confirm action</p>
              </div>
              <ChevronRight className="w-5 h-5 text-gray-400 flex-shrink-0" />
            </div>
          </div>
        </Link>

        <Link to="/error">
          <div className="bg-white rounded-2xl p-5 shadow-sm border border-gray-100 active:scale-[0.98] transition-transform">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-red-100 rounded-xl flex items-center justify-center flex-shrink-0">
                <AlertCircle className="w-6 h-6 text-red-600" />
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="font-semibold text-gray-900 mb-0.5">Error Flow</h3>
                <p className="text-xs text-gray-500">Validation and errors</p>
              </div>
              <ChevronRight className="w-5 h-5 text-gray-400 flex-shrink-0" />
            </div>
          </div>
        </Link>

        <Link to="/message">
          <div className="bg-white rounded-2xl p-5 shadow-sm border border-gray-100 active:scale-[0.98] transition-transform">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center flex-shrink-0">
                <Bell className="w-6 h-6 text-blue-600" />
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="font-semibold text-gray-900 mb-0.5">Message Flow</h3>
                <p className="text-xs text-gray-500">Toast notifications</p>
              </div>
              <ChevronRight className="w-5 h-5 text-gray-400 flex-shrink-0" />
            </div>
          </div>
        </Link>

        <Link to="/loading">
          <div className="bg-white rounded-2xl p-5 shadow-sm border border-gray-100 active:scale-[0.98] transition-transform">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center flex-shrink-0">
                <RefreshCw className="w-6 h-6 text-purple-600" />
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="font-semibold text-gray-900 mb-0.5">Loading Flow</h3>
                <p className="text-xs text-gray-500">Pull to refresh</p>
              </div>
              <ChevronRight className="w-5 h-5 text-gray-400 flex-shrink-0" />
            </div>
          </div>
        </Link>
      </div>

      <div className="mt-8 p-4 bg-blue-50 rounded-xl border border-blue-100">
        <p className="text-xs text-blue-900 leading-relaxed">
          <span className="font-semibold">Tip:</span> Each flow demonstrates a different interaction pattern with clear visual feedback.
        </p>
      </div>
    </div>
  );
}