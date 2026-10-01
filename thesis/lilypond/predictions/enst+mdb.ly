#(ly:set-option 'crop #t)
\version "2.24.4"
\header { 
tagline = " "
}
\paper {
    line-width = 150\mm 
}

\drums {
    \numericTimeSignature
    \time 4/4
    <<{
        \partial 8*7 \repeat unfold 7 hh8 | \repeat unfold 6 hh8 s4
    }
    \\ {
        r8 sn8 bd8 r16 sn16 bd8 sn8 r16 sn16 bd16 sn16 <<bd8 sn8>> sn8 sn8 bd16 sn16 bd16 sn16 sn16 sn16 <<bd8 toml8>>
    }>>
}