"""Dream visualisation: decoded dream frames with actions, rewards, curiosity, survival and advantages. The GPU part
(decoding, heads) runs here; the drawing is dream_render's (matplotlib only, so it can run in a worker process)."""
import os

import numpy as np
import torch

from .dream_render import render_dream, render_dreams_pdf
from .model import PixelDecoder


@torch.no_grad()
def dream_data(agent, states, rewards, actions, combined_advantages=None, label=None, max_advantage=None):
    """One dream -> numpy data for dream_render: decoded frames, grounding-head level / tier, curiosity, p(alive)."""
    st = states.to(agent.device).float()
    zf = st[:, agent.tssm_dim:]
    with agent._amp():
        frames = PixelDecoder.to_image(agent.tokenizer.decode(zf)).cpu()
        lvl, tier = agent.tokenizer.heads.predict(zf)                  # grounding heads read z only
    curiosities = agent.two_hot.decode(agent.curiosity_head(st)).squeeze(-1).cpu()
    alive = agent.alive_prob(st).cpu()                                 # as the dreams read it
    to_np = lambda x: x.detach().float().cpu().numpy()                 # noqa: E731
    return {"frames": np.clip(frames.float().numpy(), 0, 1), "level": lvl.cpu().numpy(), "tier": tier.cpu().numpy(),
            "rewards": to_np(rewards), "actions": torch.as_tensor(actions).argmax(-1).cpu().numpy(),
            "curiosity": curiosities.numpy(), "alive": alive.numpy(),
            "adv": to_np(combined_advantages) if combined_advantages is not None else None,
            "label": label, "max_advantage": None if max_advantage is None else float(max_advantage)}


def plot_dream(agent, states, rewards, actions, combined_advantages=None, label=None, max_advantage=None, pdf=None):
    """Draws one dream (decoded frames + per-step info); adds it to `pdf` or shows it."""
    render_dream(dream_data(agent, states, rewards, actions, combined_advantages, label, max_advantage), pdf=pdf)


def save_dreams_pdf(agent, dreams, directory, keep=10, executor=None):
    """Writes packed dreams to <directory>/dream_g<updates>.pdf, keeping the newest `keep` PDFs -> path. executor: the
    drawing is submitted to it (a worker process) and the PDF appears a few seconds later; the future is stored in
    save_dreams_pdf.pending."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"dream_g{agent.total_num_updates}.pdf")
    data = [dream_data(agent, states, rewards, actions, combined_advantages=adv_comb, label=label,
                       max_advantage=metric)
            for metric, states, rewards, _values, actions, _adv_r, _adv_c, adv_comb, label in dreams]
    if executor is None:
        return render_dreams_pdf(path, data, directory, keep)
    save_dreams_pdf.pending = executor.submit(render_dreams_pdf, path, data, directory, keep)
    return path


save_dreams_pdf.pending = None
