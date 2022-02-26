#!/bin/bash

if [ -e env ]; then
    PYTHON=env/bin/python
else
    PYTHON=python
fi

function shutdown() {
  tput cnorm # reset cursor
}
trap shutdown EXIT

function cursorBack() {
  echo -en "\033[$1D"
}

function spinner() {
  # make sure we use non-unicode character type locale
  # (that way it works for any locale as long as the font supports the characters)
  local LC_CTYPE=C

  local pid=$1 # Process Id of the previous running command

  echo "running $2..."
  case $(($RANDOM % 12)) in
  0)
    local spin='⠁⠂⠄⡀⢀⠠⠐⠈'
    local charwidth=3
    ;;
  1)
    local spin='-\|/'
    local charwidth=1
    ;;
  2)
    local spin="▁▂▃▄▅▆▇█▇▆▅▄▃▂▁"
    local charwidth=3
    ;;
  3)
    local spin="▉▊▋▌▍▎▏▎▍▌▋▊▉"
    local charwidth=3
    ;;
  4)
    local spin='←↖↑↗→↘↓↙'
    local charwidth=3
    ;;
  5)
    local spin='▖▘▝▗'
    local charwidth=3
    ;;
  6)
    local spin='┤┘┴└├┌┬┐'
    local charwidth=3
    ;;
  7)
    local spin='◢◣◤◥'
    local charwidth=3
    ;;
  8)
    local spin='◰◳◲◱'
    local charwidth=3
    ;;
  9)
    local spin='◴◷◶◵'
    local charwidth=3
    ;;
  10)
    local spin='◐◓◑◒'
    local charwidth=3
    ;;
  11)
    local spin='⣾⣽⣻⢿⡿⣟⣯⣷'
    local charwidth=3
    ;;
  esac

  local i=0
  tput civis # cursor invisible
  while kill -0 $pid 2>/dev/null; do
    local i=$(((i + $charwidth) % ${#spin}))
    printf "%s" "${spin:$i:$charwidth}"

    cursorBack 1
    sleep .1
  done
  tput cnorm
  wait $pid # capture exit code
  return $?
}

COMMANDS=(
    'mypy experiments scripts src'
    'flake8 experiments scripts src'
    'isort -c experiments scripts src'
    'pytest tests')
for CMD in "${COMMANDS[@]}"
do
    $PYTHON -m $CMD >presubmit.log 2>&1 &
    spinner $! $(echo "$CMD" | awk '{print $1;}')
    if [ $? -ne 0 ]; then
        echo "$(tput setaf 1)failed!$(tput sgr 0)"
        cat presubmit.log
        rm presubmit.log
        exit 1
    fi
done

rm presubmit.log
echo "$(tput setaf 2)passed!$(tput sgr 0)"
