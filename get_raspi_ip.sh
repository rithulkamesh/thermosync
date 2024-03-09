px-lan-scan () {
    LOCAL_MASK=$(ip -o -4 addr show | awk -F '[ /]+' '/global/ {print $4}' | cut -d. -f1,2,3)
    GATEWAY=$(route -n | \grep '^0.0.0.0' | awk '{print $2}')
    if [ $1 ] ; then range=$1 ; else range="10" ; fi

    for num in $(seq 1 ${range}) ; do
        IP=$LOCAL_MASK.$num
        if [[ $IP == $GATEWAY ]] ; then MACHINE="gateway" ; else MACHINE=$(avahi-resolve-address $IP 2>/dev/null | sed -e :a -e "s/$IP//g;s/\.[^>]*$//g;s/^[ \t]*//") ; fi
        ping -c 1 $IP>/dev/null
        if [ $? -eq 0 ] ; then
            echo -e "UP    $IP \t ($MACHINE)" ; else
            echo -e "DOWN  $IP"
        fi
    done
}

