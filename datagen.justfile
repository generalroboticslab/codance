# Generate the clip pools. The paper's pools are also on the Hub, ready to train
# on (see README.md); rebuilding them takes hours of IK solves:
#
#   just multilink-rc-paper small               # the waltz pool
#   just multilink-rc-paper small false stand   # the standing pool
#   just multilink-manifest 100ml_100zw <stamp> # the manifest a config trains on
#
# Inputs are the tracked source motions under the two directories below. A tag
# names a source clip.

waltz_dir := "data/reference_motion_edits_g1_g1_waltz"
stand_dir := "data/reference_motion_stand"

# One augmented waltz clip outside an rc: the contact NPZ, the force-yielded
# clip NPZ and the adapted qpos CSV for (tag, seed), written to out.
# partner_aware=true aims the forces by the partner's dance phase.
multilink-augment tag seed="0" out="data/compliant" partner_aware="false":
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p {{out}}
    fps="$(uv run python -c "import numpy as np; print(int(np.load('{{waltz_dir}}/{{tag}}_robot.npz')['fps'].item()))")"
    csv="{{out}}/waltz_{{tag}}_multilink_adapted_seed{{seed}}.csv"
    pflag=""
    [ "{{partner_aware}}" = true ] && pflag="--partner-aware-augmentation --partner-clip {{waltz_dir}}/{{tag}}_human.npz"
    uv run python scripts/augment_multilink.py motion \
        --motion {{waltz_dir}}/{{tag}}_robot.npz \
        --mode bimanual --seed {{seed}} \
        --out "$csv" \
        --contact-out {{out}}/waltz_{{tag}}_multilink_contact_seed{{seed}}.npz $pflag
    uv run python scripts/qpos_csv_to_motion_npz.py \
        --input-file "$csv" \
        --output-file {{out}}/waltz_{{tag}}_multilink_forcefield_seed{{seed}}.npz \
        --input-fps "$fps"

# Clip manifest for an rc, the list a config reads as `motion.motions`, written
# to configs/clip_manifests/ unless out is given. case is <N>ml[_<M>zw]: N force
# clips (the lowest valid seeds) and M zero-wrench clips (seeds from 0), split
# evenly over the rc's source clips and interleaved clip by clip.
# face=original tracks the original reference (same force schedules).
# zw_share is the zero-wrench share of episodes, or uniform (every weight 1.0).
# These write the three manifests the paper configs read:
#   just multilink-manifest 100ml_100zw 20260830_comz_small_002
#   just multilink-manifest 100ml_100zw 20260830_comz_small_002 original  # stiff
#   just multilink-manifest 100ml_100zw 20260830_stand_comz_small_002
multilink-manifest case rc face="adapted" zw_share="uniform" out="":
    #!/usr/bin/env bash
    set -euo pipefail
    CASE={{case}} RC={{rc}} FACE={{face}} ZW={{zw_share}} OUT={{out}} uv run python - <<'PY'
    import os
    import re
    import yaml
    case, rc, face, zw = (os.environ[k] for k in ("CASE", "RC", "FACE", "ZW"))
    assert face in ("adapted", "original"), f"unknown face: {face!r}"
    rcdir = f"data/compliant/rc/{rc}"
    meta = yaml.safe_load(open(f"{rcdir}/rc.yaml"))
    family, src, names = meta["family"], meta["src_dir"], meta["clip_names"]
    solo = family != "waltz"  # solo families have no partner clip
    tags = list(names)
    m = re.fullmatch(r"(\d+)ml(?:_(\d+)zw)?", case)
    assert m, f"case must be <N>ml[_<M>zw], got {case!r}"
    total_ml, total_zw = int(m.group(1)), int(m.group(2) or 0)
    assert total_ml % len(tags) == 0 and total_zw % len(tags) == 0, (
      f"case {case}: counts must split evenly over {len(tags)} source clips"
    )
    n, n_zw = total_ml // len(tags), total_zw // len(tags)
    def ml_seeds(tag):
      lst = sorted(int(x) for x in meta["valid_seeds"].get(tag, []))
      assert len(lst) >= n, f"rc {rc}: {tag} has {len(lst)} valid seeds, case needs {n}"
      return lst[:n]
    def interleaved(seeds_of):
      return [p for row in zip(*([(t, s) for s in seeds_of(t)] for t in tags)) for p in row]
    def entry(name, tag, robot, contact, weight):
      e = {
        "name": name,
        "robot_motion_file": robot,
        "free_motion_file": f"{src}/{tag}_robot.npz",
        "contact_file": contact,
      }
      if not solo:
        e["human_motion_file"] = f"{src}/{tag}_human.npz"
        e["placement_offset"] = [-0.5, 0.0, 0.0]
        e["foot_follow_offset"] = [-0.5, 0.0]
      e["weight"] = weight
      e["trackable"] = True
      e["amp_group"] = "forcefield"
      return e
    entries = [
      entry(
        f"ml_{names[tag]}_s{s}", tag,
        f"{src}/{tag}_robot.npz" if face == "original"
        else f"{rcdir}/{family}_{tag}_multilink_forcefield_seed{s}.npz",
        f"{rcdir}/{family}_{tag}_multilink_contact_seed{s}.npz",
        1.0,
      )
      for tag, s in interleaved(ml_seeds)
    ]
    if n_zw:
      w = 1.0 if zw == "uniform" else round(float(zw) / (1.0 - float(zw)) * n / n_zw, 4)
      entries += [
        entry(
          f"zw_{names[tag]}_s{s}", tag,
          f"{src}/{tag}_robot.npz",
          f"{rcdir}/{family}_{tag}_zerowrench_contact_seed{s}.npz",
          w,
        )
        for tag, s in interleaved(lambda tag: range(n_zw))
      ]
    missing = [
      p for e in entries
      for k, p in e.items()
      if k.endswith("_file") and not os.path.exists(p)
    ]
    assert not missing, f"missing rc products: {missing[:4]}"
    stem = f"multilink_{family}_{case}_{rc}" + ("_track_original" if face == "original" else "")
    out = os.environ["OUT"] or f"configs/clip_manifests/{stem}.yaml"
    with open(out, "w") as f:
      f.write(f"# Clip manifest {stem}: {len(entries)} entries, written by\n")
      f.write("# `just multilink-manifest`. A config reads it as `motion.motions: <this path>`.\n")
      yaml.safe_dump(entries, f, sort_keys=False)
    print(f"wrote {out}: {len(entries)} entries")
    PY

# One zero-wrench contact NPZ for a waltz source clip, outside an rc: zero
# force, robot stiffness held at random values redrawn every 2 to 5 s.
# mode=bimanual redraws all channels together, as the force clips switch;
# independent redraws each channel on its own. with_torque=true also varies
# the rotational stiffness (pair it with torque force clips).
multilink-zerowrench tag out="data/compliant" seed="0" mode="bimanual" with_torque="false":
    uv run python scripts/augment_multilink.py zerowrench \
        --motion {{waltz_dir}}/{{tag}}_robot.npz --seed {{seed}} --mode {{mode}} \
        {{ if with_torque == "true" { "--with-torque" } else { "" } }} \
        --out {{out}}/waltz_{{tag}}_zerowrench_contact_seed{{seed}}.npz

# One rc (release-candidate) batch in data/compliant/rc/<stamp>/: per source
# clip, `seeds` force clips (contact + clip NPZ) and `zw_seeds` zero-wrench
# contact NPZs, plus rc.yaml (settings, valid and rejected seeds, the command)
# and intermediates under work/. A seed whose forces all fall back to zero is
# rejected and the next seed fills in. Seeds are deterministic, so resume=true
# extends an existing rc exactly as a fresh build would (keep the settings).
#   stamp          folder name; empty picks the next free <today>_NNN
#   jobs           parallel seed workers
#   with_torque    bounded torques in the force clips, varying rotational
#                  stiffness in the zero-wrench clips
#   partner_aware  aim the forces by the partner's dance phase (waltz only)
#   family         waltz (paired, two source clips) or stand (solo)
#   tags, src_dir  override the family's source clips and their directory
#   extra          more flags for `scripts/augment_multilink.py motion`
multilink-rc seeds="50" stamp="" jobs="4" zw_seeds="1" with_torque="false" resume="false" partner_aware="false" family="waltz" tags="" src_dir="" extra="":
    #!/usr/bin/env bash
    set -euo pipefail
    stamp="{{stamp}}"
    if [ -z "$stamp" ]; then
        d="$(date +%Y%m%d)"
        n=1
        while [ -e "data/compliant/rc/${d}_$(printf '%03d' "$n")" ]; do n=$((n+1)); done
        stamp="${d}_$(printf '%03d' "$n")"
    fi
    rc="data/compliant/rc/${stamp}"
    if [ "{{resume}}" != true ]; then
        [ ! -e "$rc" ] || { echo "rc exists: $rc (rc folders are immutable; pick a new stamp, or pass resume=true to extend it in place)" >&2; exit 1; }
    fi
    family="{{family}}"
    src="{{src_dir}}"
    tags="{{tags}}"
    if [ "$family" = waltz ]; then
        [ -n "$src" ] || src="{{waltz_dir}}"
        [ -n "$tags" ] || tags="20260224_001 20260408_001"
    else
        [ -n "$src" ] || src="{{stand_dir}}"
        [ -n "$tags" ] || tags="20260814_001"
        [ "{{partner_aware}}" != true ] || { echo "partner_aware needs a paired family (no human clip in family $family)" >&2; exit 1; }
    fi
    for tag in $tags; do
        [ -f "$src/${tag}_robot.npz" ] || { echo "missing precondition: $src/${tag}_robot.npz" >&2; exit 1; }
        if [ "$family" = waltz ]; then
            [ -f "$src/${tag}_human.npz" ] || { echo "missing precondition: $src/${tag}_human.npz" >&2; exit 1; }
        fi
    done
    mkdir -p "$rc/work/logs"
    created="$(date +%Y-%m-%dT%H:%M:%S%:z)"
    tflag=""
    [ "{{with_torque}}" = true ] && tflag="--with-torque"
    for tag in $tags; do
        for zs in $(seq 0 $(({{zw_seeds}} - 1))); do
            [ -f "$rc/${family}_${tag}_zerowrench_contact_seed${zs}.npz" ] && continue
            uv run python scripts/augment_multilink.py zerowrench \
                --motion "$src/${tag}_robot.npz" --seed "$zs" --mode bimanual $tflag \
                --out "$rc/${family}_${tag}_zerowrench_contact_seed${zs}.npz"
        done
        fps="$(uv run python -c "import numpy as np; print(int(np.load('$src/${tag}_robot.npz')['fps'].item()))")"
        export "FPS_${tag}=${fps}"
    done
    export RC="$rc" SRC="$src" FAMILY="$family" TFLAG="$tflag" RESUME="{{resume}}" PARTNER="{{partner_aware}}"
    export EXTRA="{{extra}}"
    : > "$rc/work/rejected_seeds.txt"
    cat > "$rc/work/worker.sh" <<'WORKER'
    #!/usr/bin/env bash
    set -euo pipefail
    tag=$1; seed=$2
    log="$RC/work/logs/${tag}_s${seed}.log"
    adapted="$RC/work/${FAMILY}_${tag}_multilink_adapted_seed${seed}.csv"
    contact="$RC/${FAMILY}_${tag}_multilink_contact_seed${seed}.npz"
    clip="$RC/${FAMILY}_${tag}_multilink_forcefield_seed${seed}.npz"
    if [ "${RESUME:-false}" = true ] && [ -f "$contact" ] && [ -f "$clip" ]; then
        echo "skip $tag $seed (products exist)"
        exit 0
    fi
    try2() { "$@" || "$@"; }
    pflag=""
    [ "${PARTNER:-false}" = true ] && pflag="--partner-aware-augmentation --partner-clip $SRC/${tag}_human.npz"
    aug() { uv run python scripts/augment_multilink.py motion \
        --motion "$SRC/${tag}_robot.npz" --mode bimanual --seed "$seed" $TFLAG $pflag \
        ${EXTRA:-} --out "$adapted" --contact-out "$contact"; }
    try2 aug >"$log" 2>&1
    events="$(uv run python -c "import numpy as np; print(len(np.load('$contact')['event_f_start']))")"
    if [ "$events" -eq 0 ]; then
        rm -f "$adapted" "${adapted%.csv}_labeled.csv" "$contact"
        echo "$tag $seed" >> "$RC/work/rejected_seeds.txt"
        echo "reject $tag $seed (0 surviving events)"
        exit 0
    fi
    eval "fps=\$FPS_${tag}"
    try2 uv run python scripts/qpos_csv_to_motion_npz.py \
        --input-file "$adapted" --output-file "$clip" \
        --input-fps "$fps" >>"$log" 2>&1
    try2 uv run python scripts/multilink_table_csv.py \
        --adapted "$adapted" --contact "$contact" \
        --out "$RC/work/${FAMILY}_${tag}_multilink_table_seed${seed}.csv" >>"$log" 2>&1
    echo "done $tag $seed ($events events)"
    WORKER
    chmod +x "$rc/work/worker.sh"
    for tag in $tags; do for s in $(seq 0 $(({{seeds}} - 1))); do echo "$tag $s"; done; done | \
        xargs -P {{jobs}} -L 1 bash "$rc/work/worker.sh"
    for tag in $tags; do
        next={{seeds}}
        while :; do
            valid="$(ls "$rc"/${family}_${tag}_multilink_contact_seed*.npz 2>/dev/null | wc -l)"
            [ "$valid" -ge {{seeds}} ] && break
            [ "$next" -lt $(({{seeds}} * 3)) ] || { echo "too many rejected seeds for $tag" >&2; exit 1; }
            need=$(({{seeds}} - valid))
            for s in $(seq "$next" $((next + need - 1))); do echo "$tag $s"; done | \
                xargs -P {{jobs}} -L 1 bash "$rc/work/worker.sh"
            next=$((next + need))
        done
    done
    missing=0
    for tag in $tags; do
        n_zw="$(ls "$rc"/${family}_${tag}_zerowrench_contact_seed*.npz 2>/dev/null | wc -l)"
        [ "$n_zw" -ge {{zw_seeds}} ] || { echo "MISSING zw $tag: $n_zw < {{zw_seeds}}"; missing=$((missing + 1)); }
        n_contact="$(ls "$rc"/${family}_${tag}_multilink_contact_seed*.npz 2>/dev/null | wc -l)"
        n_clip="$(ls "$rc"/${family}_${tag}_multilink_forcefield_seed*.npz 2>/dev/null | wc -l)"
        checks=("contact $n_contact" "forcefield $n_clip")
        for pair in "${checks[@]}"; do
            set -- $pair
            [ "$2" -ge {{seeds}} ] || { echo "INCOMPLETE $tag: $1 = $2 < {{seeds}}"; missing=$((missing + 1)); }
        done
    done
    [ "$missing" -eq 0 ] || { echo "rc INCOMPLETE ($missing problems); see $rc/work/logs" >&2; exit 1; }
    {
        echo "stamp: '$stamp'"
        echo "created: '$created'"
        echo "git_commit: $(git rev-parse HEAD)"
        echo "git_dirty_files: $(git status --porcelain | wc -l)"
        echo "family: $family"
        echo "src_dir: $src"
        tags_yaml=""
        names_yaml=""
        for tag in $tags; do
            tags_yaml="${tags_yaml}'${tag}', "
            if [ "$family" = waltz ] && [ "$tag" = 20260224_001 ]; then name=human_fwd
            elif [ "$family" = waltz ] && [ "$tag" = 20260408_001 ]; then name=human_rev
            else name="$family"; fi
            names_yaml="${names_yaml}'${tag}': ${name}, "
        done
        echo "tags: [${tags_yaml%, }]"
        echo "clip_names: {${names_yaml%, }}"
        echo "seeds: {{seeds}}"
        echo "zw_seeds: {{zw_seeds}}"
        echo "with_torque: {{with_torque}}"
        echo "partner_aware: {{partner_aware}}"
        echo "extra: '{{extra}}'"
        echo "valid_seeds:"
        for tag in $tags; do
            lst="$(ls "$rc"/${family}_${tag}_multilink_contact_seed*.npz | sed -E 's/.*_seed([0-9]+)\.npz/\1/' | sort -n | paste -sd, -)"
            echo "  '${tag}': [${lst}]"
        done
        echo "rejected_seeds:"
        for tag in $tags; do
            lst="$(grep "^${tag} " "$rc/work/rejected_seeds.txt" | awk '{print $2}' | sort -nu | paste -sd, - || true)"
            echo "  '${tag}': [${lst}]"
        done
        echo "command: just multilink-rc {{seeds}} $stamp {{jobs}} {{zw_seeds}} {{with_torque}} false {{partner_aware}} $family '$tags' $src '{{extra}}'  # deterministic per seed: reproduces this rc from scratch"
    } > "$rc/rc.yaml"
    echo "rc ready: $rc"

# The paper's pools: multilink-rc with every sampler setting spelled out.
#   comz    small: the CoM height is free (downward drags become squats);
#           large: it is pinned (drags become bows)
#   torque  true adds bounded torques (stamp suffix _003 instead of _002)
#   family  waltz: forces aimed by the partner, no counter-forces (the clip
#           has no turns); stand: unguided, every force mode
#   jobs    2 fits a 32 GB host; more RAM takes 4 to 6
#   date    the stamp's date; empty means today (the hosted pools carry 20260830,
#           so a new batch never lands in a downloaded folder)
# The stamp is <date>_comz_<comz>_002 (waltz) or <date>_stand_comz_<comz>_002.
multilink-rc-paper comz torque="false" family="waltz" seeds="200" zw_seeds="200" date="" resume="false" jobs="2":
    #!/usr/bin/env bash
    set -euo pipefail
    d="{{date}}"
    [ -n "$d" ] || d="$(date +%Y%m%d)"
    case "{{comz}}" in
        small) z=0.00001 ;;
        large) z=1.0 ;;
        *) echo "comz must be small|large (got {{comz}})" >&2; exit 1 ;;
    esac
    common="--max-load-vel 1.0 --com-cost 0.1 --com-cost-z-factor $z --max-rot-disp 0.5"
    if [ "{{family}}" = waltz ]; then
        pa=true
        extra="--single-prob 0.25 --oppose-prob 0.075 --vertical-scale 0.5 $common"
        stamp_base="${d}_comz_{{comz}}"
    else
        pa=false
        extra="--pin-free-wrists --single-prob 0.4 --oppose-prob 0.1 --counter-prob 0.1 --vertical-scale 1.0 --max-disp 0.45 --com-cap 0.15 $common"
        stamp_base="${d}_{{family}}_comz_{{comz}}"
    fi
    [ "{{torque}}" = true ] && stamp="${stamp_base}_003" || stamp="${stamp_base}_002"
    echo "$stamp extra: $extra"
    just multilink-rc {{seeds}} "$stamp" {{jobs}} {{zw_seeds}} {{torque}} {{resume}} "$pa" {{family}} "" "" "$extra"

# The standing source clip: the bent-knee default pose held for 10 s at 50 Hz.
# Stand training episodes reset to this pose, so they start on the reference.
stand-clip out="data/reference_motion_stand/20260814_001_robot.npz":
    uv run python scripts/make_stand_clip.py --output-file {{out}}
