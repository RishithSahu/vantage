/* Microphone -> 16 kHz mono 16-bit blocks (a quarter of a second each) for the on-device speech recogniser.  Runs in the audio thread; nothing leaves the computer. */
class PcmDown extends AudioWorkletProcessor {
  constructor() { super(); this.r = sampleRate / 16000; this.t = 0; this.sum = 0; this.cnt = 0; this.out = new Int16Array(4000); this.n = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0]; if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.sum += ch[i]; this.cnt++; this.t += 1;
      if (this.t >= this.r) {
        this.t -= this.r; const v = Math.max(-1, Math.min(1, this.sum / this.cnt)); this.sum = 0; this.cnt = 0;
        this.out[this.n++] = v < 0 ? v * 32768 : v * 32767;
        if (this.n === this.out.length) { this.port.postMessage(this.out.buffer.slice(0)); this.n = 0; }
      }
    }
    return true;
  }
}
registerProcessor('pcm-down', PcmDown);
