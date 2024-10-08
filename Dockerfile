FROM zheli21/pytorch:2.4.1-cp311-cuda124-2204 AS base
RUN pip install -U stable-baselines3 h5py

FROM base AS git-repos
RUN mkdir /root/.ssh/
COPY id_ed25519 /root/.ssh/id_ed25519
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
RUN git clone git@github.com:lizhe07/jarvis.git
RUN git clone -b 0.4 git@github.com:XaqLab/irc-gym.git
RUN git clone git@github.com:XaqLab/hex-arena.git

FROM base AS final
COPY --from=git-repos /jarvis /jarvis
RUN pip install /jarvis
COPY --from=git-repos /irc-gym /irc-gym
RUN pip install /irc-gym
COPY --from=git-repos /hex-arena /hex-arena
WORKDIR /hex-arena
