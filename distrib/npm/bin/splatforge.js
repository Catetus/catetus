#!/usr/bin/env node
// JS shim — execs into the native `splatforge` binary downloaded by
// install.js into ../native/. Forwards argv, stdio, and exit code.
"use strict";
require("../shim").run("splatforge");
