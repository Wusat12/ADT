#(ly:set-option 'crop #t)
\version "2.24.4"
\header { 
tagline = " "
}

\drums {
    \numericTimeSignature
    \time 4/4
    <<{
        \repeat unfold 4 hh8 cymc4
    }
    \\ {
        bd4 sn4 bd4 \repeat unfold 4 toml16
    }>>
}