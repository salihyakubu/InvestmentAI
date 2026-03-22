type MessageCallback = (data: unknown) => void;

export class WebSocketManager {
  private ws: WebSocket | null = null;
  private url: string;
  private callbacks: Map<string, Set<MessageCallback>> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private _connected = false;

  constructor(baseUrl = '') {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = baseUrl || window.location.host;
    this.url = `${protocol}//${host}/ws`;
  }

  get connected(): boolean {
    return this._connected;
  }

  connect(channel = 'default'): void {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    const token = localStorage.getItem('auth_token');
    const wsUrl = `${this.url}/${channel}${token ? `?token=${token}` : ''}`;

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      this._connected = true;
      this.reconnectAttempts = 0;
      this.notifyChannel('_connection', { status: 'connected' });
    };

    this.ws.onmessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data as string) as {
          channel?: string;
          [key: string]: unknown;
        };
        const msgChannel = data.channel ?? 'default';
        this.notifyChannel(msgChannel, data);
        this.notifyChannel('*', data);
      } catch {
        // ignore malformed messages
      }
    };

    this.ws.onclose = () => {
      this._connected = false;
      this.notifyChannel('_connection', { status: 'disconnected' });
      this.attemptReconnect(channel);
    };

    this.ws.onerror = () => {
      this._connected = false;
    };
  }

  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.reconnectAttempts = this.maxReconnectAttempts;
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this._connected = false;
  }

  onMessage(channel: string, callback: MessageCallback): () => void {
    if (!this.callbacks.has(channel)) {
      this.callbacks.set(channel, new Set());
    }
    this.callbacks.get(channel)!.add(callback);

    return () => {
      this.callbacks.get(channel)?.delete(callback);
    };
  }

  send(data: unknown): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  private notifyChannel(channel: string, data: unknown): void {
    this.callbacks.get(channel)?.forEach((cb) => cb(data));
  }

  private attemptReconnect(channel: string): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) return;

    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30_000);
    this.reconnectAttempts++;

    this.reconnectTimer = setTimeout(() => {
      this.connect(channel);
    }, delay);
  }
}

export const wsManager = new WebSocketManager();
