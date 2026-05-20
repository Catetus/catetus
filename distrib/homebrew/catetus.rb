# Homebrew formula for the Catetus CLI.
#
# This file lives in the main Catetus repo so that the formula is
# versioned alongside the source it installs. The release-runbook
# (`distrib/RELEASE.md`) explains how to copy it into the public
# `catetus/homebrew-tap` repo on each release.
#
# After `gh repo create catetus/homebrew-tap --public` and the first
# successful release, users install via:
#
#     brew tap catetus/tap
#     brew install catetus
#
# Both architectures (arm64 and x86_64) of macOS are supported. Linux is
# also supported via the same archives we publish for the npm postinstall;
# linuxbrew users get the same binary the Linux .tar.gz contains.
#
# IMPORTANT: the `sha256` values below MUST be regenerated on every release.
# `distrib/RELEASE.md` documents the exact `sha256sum` command to run, and
# `scripts/release/update-homebrew-formula.sh` automates the rewrite.
# Leaving stale SHAs here is the single largest distribution-risk for the
# project — `brew install` fails loudly with `SHA256 mismatch` and users
# assume the project is broken.

class Catetus < Formula
  desc "Deterministic Gaussian-splat optimizer, validator, and converter"
  homepage "https://github.com/catetus/catetus"
  license "Apache-2.0"
  # Bumped automatically by scripts/release/update-homebrew-formula.sh.
  version "0.1.0"

  # The base URL where the release workflow uploads archives. The actual
  # archive filename for each arch is appended in the on-arch blocks below.
  # Pattern: catetus-v<version>-<target>.tar.gz
  @@base_url = "https://github.com/catetus/catetus/releases/download/v#{version}"

  on_macos do
    on_arm do
      url     "#{@@base_url}/catetus-v#{version}-aarch64-apple-darwin.tar.gz"
      # 64-zero placeholder — overwritten on release by the runbook script.
      sha256  "0000000000000000000000000000000000000000000000000000000000000000"
    end

    on_intel do
      url     "#{@@base_url}/catetus-v#{version}-x86_64-apple-darwin.tar.gz"
      sha256  "0000000000000000000000000000000000000000000000000000000000000000"
    end
  end

  on_linux do
    on_intel do
      url     "#{@@base_url}/catetus-v#{version}-x86_64-unknown-linux-gnu.tar.gz"
      sha256  "0000000000000000000000000000000000000000000000000000000000000000"
    end

    on_arm do
      url     "#{@@base_url}/catetus-v#{version}-aarch64-unknown-linux-gnu.tar.gz"
      sha256  "0000000000000000000000000000000000000000000000000000000000000000"
    end
  end

  def install
    # The archive root is `catetus-v<version>-<target>/`; brew extracts
    # into that directory, so `bin.install` finds the binaries here.
    bin.install "catetus"
    bin.install "catetus-khr-validate"
    bin.install "catetus-usd-validate"
  end

  test do
    # `--version` must succeed for `brew test catetus` to pass.
    assert_match(/catetus/i, shell_output("#{bin}/catetus --version 2>&1"))
    assert_match(/catetus-khr-validate/i,
                 shell_output("#{bin}/catetus-khr-validate --help 2>&1"))
    assert_match(/catetus-usd-validate/i,
                 shell_output("#{bin}/catetus-usd-validate --help 2>&1"))
  end
end
