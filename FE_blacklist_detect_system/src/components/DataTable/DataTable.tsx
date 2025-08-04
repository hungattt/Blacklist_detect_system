import { DataGrid, GridColDef } from "@mui/x-data-grid";
import React from "react";

type Row = {
  // id: number;
  // avatar: string;
  fullName: string;
  yearOfBirth: number;
  hometown: string;
  IdNumber: string;
  // violation: string;
  // action: React.ReactNode;
};

type DataTableProps = {
  rows: Row[];
  columns: GridColDef[];
  loading: boolean;
  rowCount: number;
  paginationModel: {
    page: number;
    pageSize: number;
  };
  setPaginationModel: React.Dispatch<
    React.SetStateAction<{
      page: number;
      pageSize: number;
    }>
  >;
};

const DataTable = ({
  columns,
  rows,
  loading,
  rowCount,
  paginationModel,
  setPaginationModel,
}: DataTableProps) => {
  return (
    <>
      {rows && (
        <div
          className="absolute z-1 w-[75%] h-[80%] top-1/2 left-1/2 
        transform -translate-x-1/2 -translate-y-1/2"
        >
          <DataGrid
            className="w-full h-full"
            columns={columns}
            rows={rows}
            rowCount={rowCount}
            loading={loading}
            pageSizeOptions={[5, 10]}
            paginationModel={paginationModel}
            paginationMode="server"
            onPaginationModelChange={(newPage) =>
              setPaginationModel((prev) => ({
                ...prev,
                page: newPage.page,
                pageSize: newPage.pageSize,
              }))
            }
          />
        </div>
      )}
    </>
  );
};

export default DataTable;
