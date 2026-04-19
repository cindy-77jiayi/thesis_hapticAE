import { useCallback, useEffect, useRef, useState } from "react";

import { SERIAL_BAUD_RATE } from "../config";

type SerialStatus = "unsupported" | "disconnected" | "connecting" | "connected";

interface SerialState {
  isSupported: boolean;
  status: SerialStatus;
  error: string | null;
  portLabel: string | null;
  connect: () => Promise<boolean>;
  disconnect: () => Promise<void>;
  sendStimulus: (stimulusId: number) => Promise<boolean>;
}

async function closePort(port: SerialPort | null): Promise<void> {
  if (!port) {
    return;
  }

  try {
    await port.close();
  } catch {
    // Ignore close errors so the UI can recover gracefully.
  }
}

export function useSerialConnection(): SerialState {
  const isSupported = typeof navigator !== "undefined" && "serial" in navigator;
  const [status, setStatus] = useState<SerialStatus>(isSupported ? "disconnected" : "unsupported");
  const [error, setError] = useState<string | null>(null);
  const [portLabel, setPortLabel] = useState<string | null>(null);
  const portRef = useRef<SerialPort | null>(null);

  const describePort = useCallback((port: SerialPort) => {
    const info = port.getInfo();
    const vendor = info.usbVendorId ? `VID ${info.usbVendorId.toString(16)}` : "Serial device";
    const product = info.usbProductId ? ` / PID ${info.usbProductId.toString(16)}` : "";
    return `${vendor}${product}`.toUpperCase();
  }, []);

  const openPort = useCallback(
    async (port: SerialPort) => {
      await closePort(portRef.current);
      await port.open({
        baudRate: SERIAL_BAUD_RATE,
        dataBits: 8,
        stopBits: 1,
        parity: "none",
        flowControl: "none",
      });
      portRef.current = port;
      setPortLabel(describePort(port));
      setStatus("connected");
      setError(null);
    },
    [describePort],
  );

  const connect = useCallback(async () => {
    if (!isSupported) {
      setError("This browser does not support Web Serial. Use the latest Chrome or Edge.");
      return false;
    }

    setStatus("connecting");
    setError(null);

    try {
      const port = await navigator.serial.requestPort();
      await openPort(port);
      return true;
    } catch (connectionError) {
      setStatus("disconnected");
      setError(connectionError instanceof Error ? connectionError.message : "Failed to connect to the ESP32.");
      return false;
    }
  }, [isSupported, openPort]);

  const disconnect = useCallback(async () => {
    await closePort(portRef.current);
    portRef.current = null;
    setPortLabel(null);
    setStatus(isSupported ? "disconnected" : "unsupported");
  }, [isSupported]);

  const sendStimulus = useCallback(
    async (stimulusId: number) => {
      const port = portRef.current;

      if (!port || status !== "connected" || !port.writable) {
        setError("Connect the ESP32 before triggering a vibration.");
        return false;
      }

      const writer = port.writable.getWriter();
      const encoder = new TextEncoder();

      try {
        await writer.write(encoder.encode(`${stimulusId}\n`));
        setError(null);
        return true;
      } catch (sendError) {
        setError(
          sendError instanceof Error ? sendError.message : "Failed to send the anonymous vibration ID to the ESP32.",
        );
        setStatus("disconnected");
        return false;
      } finally {
        writer.releaseLock();
      }
    },
    [status],
  );

  useEffect(() => {
    if (!isSupported) {
      return undefined;
    }

    let isActive = true;

    navigator.serial.getPorts().then(async (ports) => {
      if (!isActive || ports.length === 0) {
        return;
      }

      try {
        setStatus("connecting");
        await openPort(ports[0]);
      } catch {
        setStatus("disconnected");
      }
    });

    const handleDisconnect = async (event: Event) => {
      const portEvent = event as SerialConnectionEvent;
      if (!portRef.current || portEvent.port !== portRef.current) {
        return;
      }

      await disconnect();
      setError("The serial device disconnected.");
    };

    navigator.serial.addEventListener("disconnect", handleDisconnect);

    return () => {
      isActive = false;
      navigator.serial.removeEventListener("disconnect", handleDisconnect);
    };
  }, [disconnect, isSupported, openPort]);

  return {
    isSupported,
    status,
    error,
    portLabel,
    connect,
    disconnect,
    sendStimulus,
  };
}
