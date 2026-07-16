export type Vec2 = { x: number; y: number };

export type WorldEvent =
  | "kickoff"
  | "coast"
  | "thrust"
  | "impact"
  | "wall"
  | "goal";

export type WorldState = {
  playerPosition: Vec2;
  playerVelocity: Vec2;
  puckPosition: Vec2;
  puckVelocity: Vec2;
  score: number;
  tick: number;
  resetTimer: number;
  lastEvent: WorldEvent;
  randomState: number;
};

export const WORLD = {
  fps: 20,
  substeps: 4,
  wall: 0.045,
  playerRadius: 0.064,
  puckRadius: 0.043,
  playerMass: 1.8,
  puckMass: 1,
  playerAcceleration: 3.35,
  playerDrag: 1.55,
  puckDrag: 0.12,
  maxPlayerSpeed: 1.25,
  restitution: 0.91,
  goalLow: 0.35,
  goalHigh: 0.65,
  goalPauseSteps: 7,
} as const;

export const ACTION_VECTORS: ReadonlyArray<Vec2> = [
  { x: 0, y: 0 },
  { x: 0, y: -1 },
  { x: 1, y: -1 },
  { x: 1, y: 0 },
  { x: 1, y: 1 },
  { x: 0, y: 1 },
  { x: -1, y: 1 },
  { x: -1, y: 0 },
  { x: -1, y: -1 },
];

export const ACTION_NAMES = [
  "coast",
  "up",
  "up-right",
  "right",
  "down-right",
  "down",
  "down-left",
  "left",
  "up-left",
] as const;

function random(state: WorldState) {
  state.randomState = (state.randomState + 0x6d2b79f5) | 0;
  let value = state.randomState;
  value = Math.imul(value ^ (value >>> 15), value | 1);
  value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
  return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
}

function jitter(state: WorldState, amount: number) {
  return (random(state) * 2 - 1) * amount;
}

export function createWorld(seed = 7): WorldState {
  const state: WorldState = {
    playerPosition: { x: 0.27, y: 0.5 },
    playerVelocity: { x: 0, y: 0 },
    puckPosition: { x: 0.57, y: 0.5 },
    puckVelocity: { x: 0, y: 0 },
    score: 0,
    tick: 0,
    resetTimer: 0,
    lastEvent: "kickoff",
    randomState: seed | 0,
  };
  resetRound(state);
  return state;
}

export function resetRound(state: WorldState, resetScore = false) {
  state.playerPosition = { x: 0.27 + jitter(state, 0.025), y: 0.5 + jitter(state, 0.16) };
  state.playerVelocity = { x: jitter(state, 0.06), y: jitter(state, 0.06) };
  state.puckPosition = { x: 0.57 + jitter(state, 0.045), y: 0.5 + jitter(state, 0.19) };
  state.puckVelocity = { x: jitter(state, 0.08), y: jitter(state, 0.08) };
  state.resetTimer = 0;
  state.lastEvent = "kickoff";
  if (resetScore) {
    state.score = 0;
    state.tick = 0;
  }
}

export function resetPassiveRound(state: WorldState, resetScore = false) {
  if (resetScore) state.score = 0;
  const phase = state.score % 5;
  const playerAngle = resetScore ? random(state) * Math.PI * 2 : 0.35 + phase * (Math.PI * 2 / 5);
  const puckAngle = resetScore ? playerAngle + Math.PI + (random(state) - 0.5) * 0.8 : playerAngle + Math.PI + 0.42;
  const playerSpeed = resetScore ? 0.34 + random(state) * 0.34 : 0.48;
  const puckSpeed = resetScore ? 0.24 + random(state) * 0.34 : 0.38;
  state.playerPosition = resetScore
    ? { x: 0.24 + random(state) * 0.28, y: 0.2 + random(state) * 0.6 }
    : { x: 0.27, y: 0.34 + 0.08 * phase };
  state.puckPosition = resetScore
    ? { x: 0.55 + random(state) * 0.25, y: 0.2 + random(state) * 0.6 }
    : { x: 0.58, y: 0.66 - 0.08 * phase };
  state.playerVelocity = { x: Math.cos(playerAngle) * playerSpeed, y: Math.sin(playerAngle) * playerSpeed };
  state.puckVelocity = { x: Math.cos(puckAngle) * puckSpeed, y: Math.sin(puckAngle) * puckSpeed };
  state.resetTimer = 0;
  state.lastEvent = "kickoff";
}

export function createPassiveWorld(seed = 7): WorldState {
  const state = createWorld(seed);
  resetPassiveRound(state, true);
  return state;
}

function speed(vector: Vec2) {
  return Math.hypot(vector.x, vector.y);
}

function bounceAxis(position: Vec2, velocity: Vec2, axis: "x" | "y", radius: number) {
  const low = WORLD.wall + radius;
  const high = 1 - WORLD.wall - radius;
  if (position[axis] < low) {
    position[axis] = low;
    velocity[axis] = Math.abs(velocity[axis]) * WORLD.restitution;
    return true;
  }
  if (position[axis] > high) {
    position[axis] = high;
    velocity[axis] = -Math.abs(velocity[axis]) * WORLD.restitution;
    return true;
  }
  return false;
}

function collideDiscs(state: WorldState) {
  const dx = state.puckPosition.x - state.playerPosition.x;
  const dy = state.puckPosition.y - state.playerPosition.y;
  let distance = Math.hypot(dx, dy);
  const minimum = WORLD.playerRadius + WORLD.puckRadius;
  if (distance >= minimum) return false;
  if (distance < 1e-7) distance = 1e-7;
  const nx = dx / distance;
  const ny = dy / distance;
  const inversePlayer = 1 / WORLD.playerMass;
  const inversePuck = 1 / WORLD.puckMass;
  const overlap = minimum - distance;
  const correction = overlap / (inversePlayer + inversePuck);
  state.playerPosition.x -= nx * correction * inversePlayer;
  state.playerPosition.y -= ny * correction * inversePlayer;
  state.puckPosition.x += nx * correction * inversePuck;
  state.puckPosition.y += ny * correction * inversePuck;

  const closing =
    (state.puckVelocity.x - state.playerVelocity.x) * nx +
    (state.puckVelocity.y - state.playerVelocity.y) * ny;
  if (closing < 0) {
    const impulse = (-(1 + WORLD.restitution) * closing) / (inversePlayer + inversePuck);
    state.playerVelocity.x -= nx * impulse * inversePlayer;
    state.playerVelocity.y -= ny * impulse * inversePlayer;
    state.puckVelocity.x += nx * impulse * inversePuck;
    state.puckVelocity.y += ny * impulse * inversePuck;
  }
  return true;
}

export function stepWorld(state: WorldState, action: number, passive = false) {
  state.tick += 1;
  state.lastEvent = "coast";
  if (state.resetTimer > 0) {
    state.resetTimer -= 1;
    state.lastEvent = "goal";
    if (state.resetTimer === 0) {
      if (passive) resetPassiveRound(state);
      else resetRound(state);
    }
    return;
  }

  const raw = ACTION_VECTORS[action] ?? ACTION_VECTORS[0];
  const magnitude = Math.hypot(raw.x, raw.y) || 1;
  const direction = { x: raw.x / magnitude, y: raw.y / magnitude };
  if (!passive && action !== 0) state.lastEvent = "thrust";
  const dt = 1 / WORLD.fps / WORLD.substeps;

  for (let substep = 0; substep < WORLD.substeps; substep += 1) {
    if (!passive) {
      state.playerVelocity.x += direction.x * WORLD.playerAcceleration * dt;
      state.playerVelocity.y += direction.y * WORLD.playerAcceleration * dt;
    }
    const playerDrag = Math.exp(-(passive ? WORLD.puckDrag : WORLD.playerDrag) * dt);
    const puckDrag = Math.exp(-WORLD.puckDrag * dt);
    state.playerVelocity.x *= playerDrag;
    state.playerVelocity.y *= playerDrag;
    state.puckVelocity.x *= puckDrag;
    state.puckVelocity.y *= puckDrag;
    const playerSpeed = speed(state.playerVelocity);
    if (playerSpeed > WORLD.maxPlayerSpeed) {
      state.playerVelocity.x *= WORLD.maxPlayerSpeed / playerSpeed;
      state.playerVelocity.y *= WORLD.maxPlayerSpeed / playerSpeed;
    }
    state.playerPosition.x += state.playerVelocity.x * dt;
    state.playerPosition.y += state.playerVelocity.y * dt;
    state.puckPosition.x += state.puckVelocity.x * dt;
    state.puckPosition.y += state.puckVelocity.y * dt;

    if (collideDiscs(state)) state.lastEvent = "impact";
    if (
      bounceAxis(state.playerPosition, state.playerVelocity, "x", WORLD.playerRadius) ||
      bounceAxis(state.playerPosition, state.playerVelocity, "y", WORLD.playerRadius)
    ) {
      state.lastEvent = "wall";
    }

    const insideGoal = state.puckPosition.y >= WORLD.goalLow && state.puckPosition.y <= WORLD.goalHigh;
    if (
      insideGoal &&
      state.puckPosition.x + WORLD.puckRadius >= 1 - WORLD.wall &&
      state.puckVelocity.x > 0
    ) {
      state.score += 1;
      state.resetTimer = WORLD.goalPauseSteps;
      state.lastEvent = "goal";
      break;
    }

    if (bounceAxis(state.puckPosition, state.puckVelocity, "y", WORLD.puckRadius)) {
      state.lastEvent = "wall";
    }
    const left = WORLD.wall + WORLD.puckRadius;
    const right = 1 - WORLD.wall - WORLD.puckRadius;
    if (state.puckPosition.x < left) {
      state.puckPosition.x = left;
      state.puckVelocity.x = Math.abs(state.puckVelocity.x) * WORLD.restitution;
      state.lastEvent = "wall";
    } else if (state.puckPosition.x > right) {
      state.puckPosition.x = right;
      state.puckVelocity.x = -Math.abs(state.puckVelocity.x) * WORLD.restitution;
      state.lastEvent = "wall";
    }
  }
}

export function autopilotAction(state: WorldState) {
  if (state.resetTimer > 0) return 0;
  const goal = { x: 1, y: 0.5 };
  const goalDx = goal.x - state.puckPosition.x;
  const goalDy = goal.y - state.puckPosition.y;
  const goalDistance = Math.hypot(goalDx, goalDy) || 1;
  let target = {
    x: state.puckPosition.x - (goalDx / goalDistance) * 0.115,
    y: state.puckPosition.y - (goalDy / goalDistance) * 0.115,
  };
  if (
    Math.hypot(
      state.playerPosition.x - state.puckPosition.x,
      state.playerPosition.y - state.puckPosition.y,
    ) < 0.13
  ) {
    target = goal;
  }
  const dx = target.x - state.playerPosition.x;
  const dy = target.y - state.playerPosition.y;
  const sx = Math.abs(dx) < 0.022 ? 0 : dx > 0 ? 1 : -1;
  const sy = Math.abs(dy) < 0.022 ? 0 : dy > 0 ? 1 : -1;
  return ACTION_VECTORS.findIndex((vector) => vector.x === sx && vector.y === sy);
}

export function keyboardAction(keys: ReadonlySet<string>) {
  const x = Number(keys.has("ArrowRight") || keys.has("d")) - Number(keys.has("ArrowLeft") || keys.has("a"));
  const y = Number(keys.has("ArrowDown") || keys.has("s")) - Number(keys.has("ArrowUp") || keys.has("w"));
  const index = ACTION_VECTORS.findIndex((vector) => vector.x === x && vector.y === y);
  return index < 0 ? 0 : index;
}

export function snapshotWorld(state: WorldState): WorldState {
  return {
    ...state,
    playerPosition: { ...state.playerPosition },
    playerVelocity: { ...state.playerVelocity },
    puckPosition: { ...state.puckPosition },
    puckVelocity: { ...state.puckVelocity },
  };
}
