set root [file dirname [file normalize [info script]]]
set destination [file join $root build vivado]
if {[info exists ::env(DEHAZE_VIVADO_BUILD_DIR)]} {set destination $::env(DEHAZE_VIVADO_BUILD_DIR)}
create_project dehaze_demo $destination -part xc7z015clg485-2 -force
set_property ip_repo_paths [list [file join $root ip]] [current_project]
update_ip_catalog -rebuild
source [file join $root hardware create_bd.tcl]
validate_bd_design
save_bd_design
if {[info exists ::env(DEHAZE_VALIDATE_ONLY)] && $::env(DEHAZE_VALIDATE_ONLY) eq "1"} {
    puts "PORTABLE_BD_VALIDATION_PASS"
    close_project
    exit
}
generate_target all [get_files design_1.bd]
set wrapper [make_wrapper -files [get_files design_1.bd] -top]
add_files -norecurse $wrapper
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1
foreach run [get_runs -filter {IS_SYNTHESIS}] {
    if {[lsearch -exact [list_property $run] STEPS.SYNTH_DESIGN.ARGS.CONTROL_SET_OPT_THRESHOLD]>=0} {
        set_property STEPS.SYNTH_DESIGN.ARGS.CONTROL_SET_OPT_THRESHOLD 16 $run
    }
}
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {![string match "*synth_design Complete*" [get_property STATUS [get_runs synth_1]]]} {error "Synthesis failed"}
open_run synth_1
report_utilization -file [file join $root logs synthesis_utilization.rpt]
set handle [open [file join $root logs synthesis_utilization.rpt] r]
set report_text [read $handle]
close $handle
if {![regexp {\| Slice LUTs\*?\s+\|\s+([0-9]+)} $report_text matched lut_count]} {error "Missing LUT utilization"}
set dsp_count [llength [get_cells -hier -filter {REF_NAME == DSP48E1}]]
puts "SPEED_SYNTH_RESOURCE LUT=$lut_count DSP=$dsp_count"
if {$lut_count>46200 || $dsp_count>160} {error "Candidate does not fit; do not deploy"}
close_design
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {![string match "*write_bitstream Complete*" [get_property STATUS [get_runs impl_1]]]} {error "Implementation failed"}
open_run impl_1
set setup [get_property SLACK [lindex [get_timing_paths -delay_type max -max_paths 1] 0]]
set hold [get_property SLACK [lindex [get_timing_paths -delay_type min -max_paths 1] 0]]
if {$setup < 0 || $hold < 0} {
    phys_opt_design -directive AggressiveExplore
    route_design -directive Explore
    set setup [get_property SLACK [lindex [get_timing_paths -delay_type max -max_paths 1] 0]]
    set hold [get_property SLACK [lindex [get_timing_paths -delay_type min -max_paths 1] 0]]
    if {$setup >= 0 && $hold >= 0} {
        write_checkpoint -force [file join $destination timing_closed.dcp]
        write_bitstream -force [file join $destination dehaze_demo.runs impl_1 design_1_wrapper.bit]
    }
}
file mkdir [file join $root logs]
report_timing_summary -file [file join $root logs timing_summary.rpt]
report_utilization -file [file join $root logs utilization.rpt]
report_drc -file [file join $root logs drc.rpt]
if {$setup < 0 || $hold < 0} {error "Timing failed setup=$setup hold=$hold"}
file mkdir [file join $root board]
write_hw_platform -fixed -include_bit -force -file [file join $root board design_1_wrapper.xsa]
puts "HD_HARDWARE_PASS clock=100MHz WNS=$setup WHS=$hold"
close_project
exit
