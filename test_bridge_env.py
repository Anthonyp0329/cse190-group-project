
import time
import mujoco.viewer as mj_viewer

from bridge_env import BridgeBuildingEnv


def main(steps=300, fps=20):
    env = BridgeBuildingEnv()
    _, _ = env.reset()

    dt = 1.0 / fps

    # Create the viewer (blocking call returns a Viewer object)
    with mj_viewer.launch_passive(env.model, env.data) as viewer:
        for _ in range(steps):
            action = env.action_space.sample()
            env.step(action)

            # Push latest physics state to the window
            viewer.sync()

            # Slow down so you can see what’s happening
            time.sleep(dt)

    env.close()


if __name__ == "__main__":
    main()