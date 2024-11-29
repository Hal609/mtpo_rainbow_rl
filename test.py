import gymnasium as gym
from smb_env.smb_env_cynes import SuperMarioBrosEnv
from SMBPreprocessingCustom import SMBPreprocessingCustom

def create_env(rom_path, framestack):
    # Register the custom environment
    gym.register(
        id="gymnasium_env/smb-v5",
        entry_point=SuperMarioBrosEnv,
    )
    # Create the environment
    return gym.wrappers.FrameStack(
        SMBPreprocessingCustom(gym.make("gymnasium_env/smb-v5", rom_path=rom_path, headless=False)),
        num_stack=framestack,
        lz4_compress=False,
    )

def main():
    rom_path = (
        "C:/Users/offan/Downloads/4398_Beyond_The_Rainbow_High_P_Supplementary Material/"
        "BeyondTheRainbowICLR/smb_env/super-mario-bros-rectangle.nes"
    )

    framestack = 1
    envs = gym.vector.AsyncVectorEnv(
        [
            lambda: create_env(rom_path, framestack) for _ in range(4)
        ],
        context="spawn",  # Required for Windows
    )
    print("Environment created successfully!")

if __name__ == "__main__":
    main()
