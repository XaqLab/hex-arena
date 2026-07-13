FROM zheli21/pytorch:2.13.0-cp314-cuda132-2404 AS base
RUN pip install -U h5py --break-system-packages

FROM base AS git-repos
COPY id_ed25519 /root/.ssh/id_ed25519
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
RUN git clone git@github.com:lizhe07/jarvis.git
RUN git clone git@github.com:lizhe07/stable-baselines3.git
RUN git clone -b 0.5 git@github.com:XaqLab/irc-gym.git
RUN git clone -b 0.2 git@github.com:XaqLab/hex-arena.git

FROM base AS final
COPY --from=git-repos /jarvis /jarvis
RUN pip install /jarvis --break-system-packages
COPY --from=git-repos /stable-baselines3 /stable-baselines3
RUN pip install /stable-baselines3 --break-system-packages
RUN pip install -U sb3-contrib --break-system-packages
COPY --from=git-repos /irc-gym /irc-gym
RUN pip install /irc-gym --break-system-packages
COPY --from=git-repos /hex-arena /hex-arena
WORKDIR /hex-arena
